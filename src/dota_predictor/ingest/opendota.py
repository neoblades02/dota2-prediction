from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from dota_predictor.ingest.base import (
    RateLimitedClient,
    RateLimitExhaustedError,
    get_sync_cursor,
    set_sync_cursor,
)
from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.storage.models import pro_match_index
from dota_predictor.storage.raw_store import save_raw_payload
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("ingest.opendota")

REQUEST_AGAIN = timedelta(hours=1)  # a parse request not answered by then is repeated
MAX_REQUESTS = 3


def _aware(dt: datetime | None) -> datetime | None:
    return dt.replace(tzinfo=UTC) if dt is not None and dt.tzinfo is None else dt


def sync_pro_match_index(
    engine: Engine,
    since_utc: datetime | None = None,
    max_pages: int | None = None,
    resume: bool = True,
    batch_size_pages: int = 10,
) -> dict[str, Any]:
    """Ingest pro matches index from OpenDota /proMatches with resumable backwards pagination."""
    provider = "opendota"
    resource = "proMatches"
    client = RateLimitedClient(provider=provider, base_url="https://api.opendota.com/api")

    cursor: str | None = None
    if resume:
        with engine.begin() as conn:
            cursor_rec = get_sync_cursor(conn, provider, resource)
            if cursor_rec and cursor_rec.get("cursor"):
                cursor = cursor_rec["cursor"]
                logger.info("Resuming proMatches sync from cursor", cursor=cursor)

    pages_fetched = 0
    total_matches_upserted = 0
    stop_reason = "completed"

    try:
        while True:
            if max_pages is not None and pages_fetched >= max_pages:
                stop_reason = "max_pages_reached"
                break

            params = {}
            if cursor:
                params["less_than_match_id"] = cursor

            endpoint = "/proMatches"
            logger.info("Fetching pro matches page", cursor=cursor, page=pages_fetched + 1)
            resp = client.get(endpoint, params=params)
            if resp.status_code != 200:
                logger.error("Failed to fetch proMatches", status=resp.status_code, body=resp.text[:200])
                stop_reason = f"http_status_{resp.status_code}"
                break

            matches = resp.json()
            if not isinstance(matches, list) or not matches:
                logger.info("No more matches returned from proMatches")
                stop_reason = "empty_page"
                break

            now = datetime.now(UTC)
            req_key = f"cursor_{cursor or 'head'}"

            with engine.begin() as conn:
                # 1. Save raw payload
                save_raw_payload(
                    conn=conn,
                    provider=provider,
                    endpoint=endpoint,
                    request_key=req_key,
                    payload=matches,
                    http_status=resp.status_code,
                    schema_version="v1",
                    fetched_at_utc=now,
                )

                # 2. Upsert shallow matches into pro_match_index
                reached_since_boundary = False
                for m in matches:
                    start_epoch = m.get("start_time")
                    start_utc = datetime.fromtimestamp(start_epoch, tz=UTC) if start_epoch else None
                    if since_utc and start_utc and start_utc < since_utc:
                        reached_since_boundary = True

                    stmt = (
                        dialect_insert(conn, pro_match_index)
                        .values(
                            match_id=m["match_id"],
                            start_time_utc=start_utc,
                            duration_s=m.get("duration"),
                            radiant_team_id=m.get("radiant_team_id"),
                            radiant_name=m.get("radiant_name"),
                            dire_team_id=m.get("dire_team_id"),
                            dire_name=m.get("dire_name"),
                            leagueid=m.get("leagueid"),
                            league_name=m.get("league_name"),
                            series_id=m.get("series_id"),
                            series_type=m.get("series_type"),
                            radiant_score=m.get("radiant_score"),
                            dire_score=m.get("dire_score"),
                            radiant_win=m.get("radiant_win"),
                            version=str(m.get("version")) if m.get("version") is not None else None,
                            detail_status="pending",
                            first_seen_utc=now,
                            last_seen_utc=now,
                        )
                        .on_conflict_do_update(
                            index_elements=["match_id"],
                            set_={
                                "last_seen_utc": now,
                                "radiant_score": m.get("radiant_score"),
                                "dire_score": m.get("dire_score"),
                                "radiant_win": m.get("radiant_win"),
                            },
                        )
                    )
                    conn.execute(stmt)
                    total_matches_upserted += 1

                # Update cursor to smallest match_id in this page
                min_match_id = min(m["match_id"] for m in matches)
                cursor = str(min_match_id)
                set_sync_cursor(
                    conn=conn,
                    provider=provider,
                    resource=resource,
                    cursor=cursor,
                    state="running",
                    success=True,
                )

            pages_fetched += 1

            if reached_since_boundary:
                logger.info("Reached historical cutoff boundary", since_utc=since_utc.isoformat())
                stop_reason = "reached_since_boundary"
                break

    except RateLimitExhaustedError as e:
        logger.warning("Stopping proMatches sync: rate limit quota reached", reason=str(e))
        stop_reason = "rate_limit_exhausted"
    except Exception as e:
        logger.error("Exception during proMatches sync", error=str(e))
        with engine.begin() as conn:
            set_sync_cursor(
                conn=conn,
                provider=provider,
                resource=resource,
                cursor=cursor,
                state="error",
                error=str(e),
                success=False,
            )
        raise
    finally:
        client.close()

    if stop_reason == "completed":
        with engine.begin() as conn:
            set_sync_cursor(
                conn=conn,
                provider=provider,
                resource=resource,
                cursor=cursor,
                state="done",
                success=True,
            )

    return {
        "pages_fetched": pages_fetched,
        "total_matches_upserted": total_matches_upserted,
        "last_cursor": cursor,
        "stop_reason": stop_reason,
    }


def sync_match_details(
    engine: Engine,
    limit: int | None = 100,
) -> dict[str, Any]:
    """Fetch pending match details from /matches/{match_id} and save raw payloads."""
    provider = "opendota"
    client = RateLimitedClient(provider=provider, base_url="https://api.opendota.com/api")

    fetched_count = 0
    unavailable_count = 0
    errors_count = 0

    with engine.begin() as conn:
        query = (
            select(pro_match_index.c.match_id)
            .where(pro_match_index.c.detail_status == "pending")
            .order_by(pro_match_index.c.start_time_utc.desc())
        )
        if limit is not None:
            query = query.limit(limit)
        match_ids = [row[0] for row in conn.execute(query).fetchall()]

    logger.info("Starting match details sync", pending_target=len(match_ids))

    try:
        for match_id in match_ids:
            endpoint = f"/matches/{match_id}"
            try:
                resp = client.get(endpoint)
                now = datetime.now(UTC)

                if resp.status_code == 200:
                    payload = resp.json()
                    with engine.begin() as conn:
                        save_raw_payload(
                            conn=conn,
                            provider=provider,
                            endpoint=endpoint,
                            request_key=str(match_id),
                            payload=payload,
                            http_status=resp.status_code,
                            schema_version="v1",
                            fetched_at_utc=now,
                        )
                        conn.execute(
                            update(pro_match_index)
                            .where(pro_match_index.c.match_id == match_id)
                            .values(detail_status="fetched", detail_error=None)
                        )
                    fetched_count += 1
                elif resp.status_code == 404:
                    with engine.begin() as conn:
                        conn.execute(
                            update(pro_match_index)
                            .where(pro_match_index.c.match_id == match_id)
                            .values(detail_status="unavailable", detail_error="HTTP 404 Not Found")
                        )
                    unavailable_count += 1
                else:
                    with engine.begin() as conn:
                        conn.execute(
                            update(pro_match_index)
                            .where(pro_match_index.c.match_id == match_id)
                            .values(
                                detail_status="error",
                                detail_error=f"HTTP {resp.status_code}: {resp.text[:100]}",
                            )
                        )
                    errors_count += 1
            except RateLimitExhaustedError:
                logger.warning("Daily rate limit quota exhausted during match details sync")
                break
            except Exception as e:
                logger.error("Error fetching match detail", match_id=match_id, error=str(e))
                with engine.begin() as conn:
                    conn.execute(
                        update(pro_match_index)
                        .where(pro_match_index.c.match_id == match_id)
                        .values(detail_status="error", detail_error=str(e))
                    )
                errors_count += 1
    finally:
        client.close()

    return {
        "targeted": len(match_ids),
        "fetched": fetched_count,
        "unavailable": unavailable_count,
        "errors": errors_count,
    }


def sync_tournament_matches(
    engine: Engine,
    tier: str = "premium",
    min_league_id: int = 14000,
    max_tournaments: int | None = None,
) -> dict[str, Any]:
    """Ingest all matches for tournaments matching tier and min_league_id from /leagues/{id}/matches."""
    provider = "opendota"
    client = RateLimitedClient(provider=provider, base_url="https://api.opendota.com/api")

    with engine.begin() as conn:
        from dota_predictor.storage.models import tournaments

        query = (
            select(tournaments.c.leagueid, tournaments.c.title)
            .where(
                tournaments.c.leagueid >= min_league_id,
                tournaments.c.leagueid < 60000,
                tournaments.c.tier == tier,
            )
            .order_by(tournaments.c.leagueid.asc())
        )
        if max_tournaments is not None:
            query = query.limit(max_tournaments)
        leagues = conn.execute(query).fetchall()

    logger.info("Found eligible tournaments to ingest matches for", count=len(leagues), tier=tier)

    total_matches = 0
    tournaments_processed = 0

    try:
        for lid, title in leagues:
            endpoint = f"/leagues/{lid}/matches"
            resp = client.get(endpoint)
            now = datetime.now(UTC)

            if resp.status_code == 200:
                matches = resp.json()
                if isinstance(matches, list) and matches:
                    with engine.begin() as conn:
                        save_raw_payload(
                            conn=conn,
                            provider=provider,
                            endpoint=endpoint,
                            request_key=str(lid),
                            payload=matches,
                            http_status=resp.status_code,
                            schema_version="v1",
                            fetched_at_utc=now,
                        )

                        for m in matches:
                            start_epoch = m.get("start_time")
                            start_utc = datetime.fromtimestamp(start_epoch, tz=UTC) if start_epoch else None

                            stmt = (
                                dialect_insert(conn, pro_match_index)
                                .values(
                                    match_id=m["match_id"],
                                    start_time_utc=start_utc,
                                    duration_s=m.get("duration"),
                                    radiant_team_id=m.get("radiant_team_id"),
                                    radiant_name=m.get("radiant_team_name") or m.get("radiant_name"),
                                    dire_team_id=m.get("dire_team_id"),
                                    dire_name=m.get("dire_team_name") or m.get("dire_name"),
                                    leagueid=lid,
                                    league_name=title,
                                    series_id=m.get("series_id"),
                                    series_type=m.get("series_type"),
                                    radiant_score=m.get("radiant_score"),
                                    dire_score=m.get("dire_score"),
                                    radiant_win=m.get("radiant_win"),
                                    version=str(m.get("version")) if m.get("version") is not None else None,
                                    detail_status="pending",
                                    first_seen_utc=now,
                                    last_seen_utc=now,
                                )
                                .on_conflict_do_update(
                                    index_elements=["match_id"],
                                    set_={
                                        "last_seen_utc": now,
                                        "leagueid": lid,
                                        "league_name": title,
                                        "radiant_win": m.get("radiant_win"),
                                    },
                                )
                            )
                            conn.execute(stmt)
                            total_matches += 1

                tournaments_processed += 1
                logger.info(
                    "Ingested tournament matches",
                    league_id=lid,
                    tournament=title,
                    matches=len(matches) if isinstance(matches, list) else 0,
                )
    except RateLimitExhaustedError:
        logger.warning("Rate limit reached during tournament ingestion")
    finally:
        client.close()

    return {
        "tournaments_processed": tournaments_processed,
        "total_matches_upserted": total_matches,
    }


def sync_reference_data(engine: Engine) -> dict[str, Any]:
    """Ingest reference datasets: /proPlayers, /teams, /leagues, /heroStats."""
    provider = "opendota"
    client = RateLimitedClient(provider=provider, base_url="https://api.opendota.com/api")
    endpoints = ["/proPlayers", "/teams", "/leagues", "/heroStats"]
    results = {}

    try:
        now = datetime.now(UTC)
        for ep in endpoints:
            logger.info("Fetching reference endpoint", endpoint=ep)
            resp = client.get(ep)
            if resp.status_code == 200:
                payload = resp.json()
                count = len(payload) if isinstance(payload, list) else 1
                with engine.begin() as conn:
                    obj_id = save_raw_payload(
                        conn=conn,
                        provider=provider,
                        endpoint=ep,
                        request_key="reference_head",
                        payload=payload,
                        http_status=resp.status_code,
                        schema_version="v1",
                        fetched_at_utc=now,
                    )
                results[ep] = {"status": "ok", "items": count, "raw_id": obj_id}
            else:
                results[ep] = {"status": "error", "code": resp.status_code}
    finally:
        client.close()

    return results


def recover_match_details(engine: Engine, rows: list[Mapping[str, Any]]) -> dict[str, int]:
    """Games fetched by id from /matches/{id}; `rows` are pro_match_index rows.

    Leagues OpenDota rates below professional never reach its pro tables, and /matches 404s on a game it has
    not stored, so a miss asks OpenDota to pull the game from Valve (POST /request) and a later cycle picks it
    up; an unanswered request is repeated after an hour, three times at most. A fetched game is stored raw,
    normalized at once and its index row completed, so series settle from it and the ratings fold it in.
    """
    client = RateLimitedClient(provider="opendota", base_url="https://api.opendota.com/api")
    out: Counter[str] = Counter()
    try:
        for row in rows:
            match_id, status = row["match_id"], row["detail_status"]
            endpoint = f"/matches/{match_id}"
            try:
                resp = client.get(endpoint)
            except RateLimitExhaustedError:
                logger.warning("OpenDota quota exhausted while recovering games", left=len(rows) - sum(out.values()))
                break
            payload = resp.json() if resp.status_code == 200 else {}
            now = datetime.now(UTC)
            with engine.begin() as conn:
                if payload.get("radiant_win") is not None:
                    raw_id = save_raw_payload(
                        conn=conn,
                        provider="opendota",
                        endpoint=endpoint,
                        request_key=str(match_id),
                        payload=payload,
                        fetched_at_utc=now,
                    )
                    normalize_match_detail(conn, payload, raw_object_id=raw_id)
                    values: dict[str, Any] = {
                        "detail_status": "fetched",
                        "detail_error": None,
                        "last_seen_utc": now,
                        "radiant_win": payload["radiant_win"],
                        "duration_s": payload.get("duration"),
                        "radiant_score": payload.get("radiant_score"),
                        "dire_score": payload.get("dire_score"),
                    }
                    for key, src in (
                        ("series_id", "series_id"),
                        ("series_type", "series_type"),
                        ("leagueid", "leagueid"),
                    ):
                        if payload.get(src) is not None:
                            values[key] = payload[src]
                    if (payload.get("league") or {}).get("name"):
                        values["league_name"] = payload["league"]["name"]
                    if payload.get("start_time"):
                        values["start_time_utc"] = datetime.fromtimestamp(payload["start_time"], tz=UTC)
                    conn.execute(update(pro_match_index).where(pro_match_index.c.match_id == match_id).values(**values))
                    out["fetched"] += 1
                    continue
                asked = (
                    int(str(row.get("detail_error") or "requests=0").rsplit("=", 1)[-1] or 0)
                    if status == "requested"
                    else 0
                )
                last = _aware(row.get("last_seen_utc")) or now
                due = status != "requested" or (asked < MAX_REQUESTS and last <= now - REQUEST_AGAIN)
                if due and client.post(f"/request/{match_id}").status_code == 200:
                    conn.execute(
                        update(pro_match_index)
                        .where(pro_match_index.c.match_id == match_id)
                        .values(detail_status="requested", detail_error=f"requests={asked + 1}", last_seen_utc=now)
                    )
                    out["requested"] += 1
                elif status == "requested":
                    out["waiting"] += 1
                else:  # refused for now; the next cycle asks again
                    out["refused"] += 1
    finally:
        client.close()
    return dict(out)
