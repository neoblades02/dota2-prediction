"""Bulk ingestion via OpenDota's public SQL endpoint (/explorer).

One explorer call returns thousands of rows, so a full pro-match backfill takes
~100 requests instead of one request per match. Payloads are stored raw like
every other source, then normalized through the same match-detail path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.engine import Engine

from dota_predictor.ingest.base import RateLimitedClient
from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.storage.models import pro_match_index, team_aliases, teams
from dota_predictor.storage.raw_store import save_raw_payload
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("ingest.explorer")

PROVIDER = "opendota"
BASE_URL = "https://api.opendota.com/api"
DEFAULT_TIERS = ("premium", "professional")


def _query(client: RateLimitedClient, sql: str) -> list[dict[str, Any]]:
    resp = client.get("/explorer", params={"sql": sql})
    resp.raise_for_status()
    body = resp.json()
    if body.get("err"):
        raise RuntimeError(f"explorer query failed: {body['err']}")
    return body.get("rows") or []


def _epoch(dt: datetime) -> int:
    if dt.tzinfo is None:  # SQLite returns naive datetimes; everything stored is UTC
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def sync_explorer_match_index(
    engine: Engine,
    tiers: tuple[str, ...] = DEFAULT_TIERS,
    since_utc: datetime = datetime(2022, 1, 1, tzinfo=UTC),
    page_size: int = 10000,
) -> dict[str, Any]:
    """Page every league match of the given tiers into pro_match_index (status pending)."""
    client = RateLimitedClient(provider=PROVIDER, base_url=BASE_URL)
    tier_sql = ",".join(f"'{t}'" for t in tiers)
    cursor = 0
    total = 0
    pages = 0
    try:
        while True:
            sql = (
                "SELECT m.match_id, m.start_time, m.duration, m.radiant_win, m.radiant_team_id, m.dire_team_id, "
                "m.radiant_score, m.dire_score, m.leagueid, l.name AS league_name, l.tier, m.series_id, m.series_type "
                "FROM matches m JOIN leagues l ON l.leagueid = m.leagueid "
                f"WHERE l.tier IN ({tier_sql}) AND m.start_time >= {_epoch(since_utc)} AND m.match_id > {cursor} "
                "AND m.radiant_team_id IS NOT NULL AND m.dire_team_id IS NOT NULL AND m.radiant_win IS NOT NULL "
                f"ORDER BY m.match_id LIMIT {page_size}"
            )
            rows = _query(client, sql)
            if not rows:
                break
            now = datetime.now(UTC)
            with engine.begin() as conn:
                save_raw_payload(
                    conn=conn,
                    provider=PROVIDER,
                    endpoint="/explorer/matches",
                    request_key=f"tiers={','.join(tiers)};after={cursor}",
                    payload=rows,
                    fetched_at_utc=now,
                )
                for m in rows:
                    values = {
                        "match_id": m["match_id"],
                        "start_time_utc": datetime.fromtimestamp(m["start_time"], tz=UTC),
                        "duration_s": m.get("duration"),
                        "radiant_team_id": m.get("radiant_team_id"),
                        "dire_team_id": m.get("dire_team_id"),
                        "leagueid": m.get("leagueid"),
                        "league_name": m.get("league_name"),
                        "series_id": m.get("series_id"),
                        "series_type": m.get("series_type"),
                        "radiant_score": m.get("radiant_score"),
                        "dire_score": m.get("dire_score"),
                        "radiant_win": m.get("radiant_win"),
                        "last_seen_utc": now,
                    }
                    stmt = dialect_insert(conn, pro_match_index).values(
                        **values, detail_status="pending", first_seen_utc=now
                    )
                    conn.execute(stmt.on_conflict_do_update(index_elements=["match_id"], set_=values))
            total += len(rows)
            pages += 1
            cursor = rows[-1]["match_id"]
            logger.info("Indexed explorer match page", page=pages, rows=len(rows), last_match_id=cursor)
            if len(rows) < page_size:
                break
    finally:
        client.close()
    return {"indexed": total, "pages": pages}


def sync_explorer_player_matches(
    engine: Engine,
    batch_size: int = 1000,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fetch hero/player slots for pending index rows and normalize them as parsed games."""
    client = RateLimitedClient(provider=PROVIDER, base_url=BASE_URL)
    with engine.begin() as conn:
        pending = (
            conn.execute(
                select(pro_match_index)
                .where(pro_match_index.c.detail_status == "pending")
                .order_by(pro_match_index.c.start_time_utc.asc())
            )
            .mappings()
            .fetchall()
        )
    if limit is not None:
        pending = pending[:limit]
    logger.info("Starting explorer player sync", pending=len(pending), batch_size=batch_size)

    fetched = 0
    unavailable = 0
    try:
        for i in range(0, len(pending), batch_size):
            chunk = pending[i : i + batch_size]
            ids = [r["match_id"] for r in chunk]
            rows = _query(
                client,
                "SELECT match_id, account_id, hero_id, player_slot FROM player_matches "
                f"WHERE match_id IN ({','.join(map(str, ids))}) ORDER BY match_id, player_slot",
            )
            by_match: dict[int, list[dict[str, Any]]] = {}
            for r in rows:
                by_match.setdefault(r["match_id"], []).append(r)

            now = datetime.now(UTC)
            with engine.begin() as conn:
                raw_id = save_raw_payload(
                    conn=conn,
                    provider=PROVIDER,
                    endpoint="/explorer/player_matches",
                    request_key=f"{ids[0]}-{ids[-1]}",
                    payload=rows,
                    fetched_at_utc=now,
                )
                for idx in chunk:
                    players = by_match.get(idx["match_id"])
                    if not players or len(players) != 10:
                        conn.execute(
                            update(pro_match_index)
                            .where(pro_match_index.c.match_id == idx["match_id"])
                            .values(detail_status="unavailable", detail_error="explorer: no player rows")
                        )
                        unavailable += 1
                        continue
                    payload = {
                        "match_id": idx["match_id"],
                        "start_time": _epoch(idx["start_time_utc"]),
                        "duration": idx["duration_s"],
                        "radiant_win": idx["radiant_win"],
                        "radiant_team_id": idx["radiant_team_id"],
                        "radiant_name": idx["radiant_name"],
                        "dire_team_id": idx["dire_team_id"],
                        "dire_name": idx["dire_name"],
                        "leagueid": idx["leagueid"],
                        "league": {"name": idx["league_name"]},
                        "players": [
                            {"account_id": p["account_id"], "hero_id": p["hero_id"], "player_slot": p["player_slot"]}
                            for p in players
                        ],
                    }
                    normalize_match_detail(conn, payload, raw_object_id=raw_id)
                    conn.execute(
                        update(pro_match_index)
                        .where(pro_match_index.c.match_id == idx["match_id"])
                        .values(detail_status="fetched_explorer", detail_error=None)
                    )
                    fetched += 1
            logger.info("Explorer player batch done", done=i + len(chunk), total=len(pending), fetched=fetched)
    finally:
        client.close()
    return {"fetched": fetched, "unavailable": unavailable, "pending_total": len(pending)}


def sync_explorer_teams(engine: Engine, batch_size: int = 500, only_placeholders: bool = False) -> dict[str, Any]:
    """Fill in canonical team names and tags from OpenDota's teams table (optionally only "Team_<id>" ones)."""
    with engine.begin() as conn:
        rows = conn.execute(select(teams.c.team_id, teams.c.canonical_name).order_by(teams.c.team_id)).all()
    team_ids = [tid for tid, name in rows if not only_placeholders or name == f"Team_{tid}"]
    if not team_ids:
        return {"teams": 0, "named": 0}
    client = RateLimitedClient(provider=PROVIDER, base_url=BASE_URL)
    updated = 0
    try:
        for i in range(0, len(team_ids), batch_size):
            ids = team_ids[i : i + batch_size]
            rows = _query(client, f"SELECT team_id, name, tag FROM teams WHERE team_id IN ({','.join(map(str, ids))})")
            now = datetime.now(UTC)
            with engine.begin() as conn:
                save_raw_payload(
                    conn=conn,
                    provider=PROVIDER,
                    endpoint="/explorer/teams",
                    request_key=f"{ids[0]}-{ids[-1]}",
                    payload=rows,
                    fetched_at_utc=now,
                )
                for r in rows:
                    name = (r.get("name") or "").strip()
                    if not name:
                        continue
                    conn.execute(
                        update(teams)
                        .where(teams.c.team_id == r["team_id"])
                        .values(canonical_name=name, tag=r.get("tag"))
                    )
                    conn.execute(
                        update(team_aliases)
                        .where(
                            team_aliases.c.provider == PROVIDER, team_aliases.c.provider_team_id == str(r["team_id"])
                        )
                        .values(name=name, last_seen_utc=now)
                    )
                    updated += 1
    finally:
        client.close()
    logger.info("Explorer team sync done", teams=len(team_ids), named=updated)
    return {"teams": len(team_ids), "named": updated}
