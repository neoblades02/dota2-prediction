"""PandaScore schedule: tournaments and matches, upcoming through finished.

Fixture ids are ``ps:<match id>``; tournament keys are ``ps:<tournament id>``.
Opponents link to our team ids through ``normalize.team_links``; an unlinked
or TBD opponent leaves its team id null and the fixture gets no prediction.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import update
from sqlalchemy.engine import Connection, Engine

from dota_predictor.ingest.base import RateLimitedClient
from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.team_links import TeamIndex, link_team
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import fixtures, schedule_tournaments
from dota_predictor.storage.raw_store import save_raw_payload
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("ingest.pandascore")

PROVIDER = "pandascore"
BASE_URL = "https://api.pandascore.co"
PER_PAGE = 100
MAX_PAGES = 20
PAST_WINDOW = timedelta(days=3)
STATUS = {
    "not_started": "upcoming",
    "postponed": "upcoming",
    "running": "running",
    "finished": "finished",
    "canceled": "canceled",
}


def enabled() -> bool:
    return bool(get_settings().pandascore_api_key)


def make_client() -> RateLimitedClient:
    return RateLimitedClient(provider=PROVIDER, base_url=BASE_URL)


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def fetch_all(client: Any, path: str, **params: Any) -> list[dict[str, Any]]:
    """Every row of a paged list endpoint, following the x-total header."""
    out: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        resp = client.get(path, params={**params, "per_page": PER_PAGE, "page": page})
        resp.raise_for_status()
        rows = resp.json()
        out += rows
        if not rows or len(out) >= int(resp.headers.get("x-total", len(out))):
            break
    return out


def upsert_tournaments(conn: Connection, rows: list[dict[str, Any]], status: str) -> set[str]:
    now = datetime.now(UTC)
    keys = set()
    for t in rows:
        key = f"ps:{t['id']}"
        keys.add(key)
        values = {
            "league_name": (t.get("league") or {}).get("name"),
            "serie_name": (t.get("serie") or {}).get("full_name"),
            "stage_name": t.get("name"),
            "tier": t.get("tier"),
            "begin_at_utc": _ts(t.get("begin_at")),
            "end_at_utc": _ts(t.get("end_at")),
            "status": status,
            "updated_at_utc": now,
        }
        conn.execute(
            dialect_insert(conn, schedule_tournaments)
            .values(tournament_key=key, **values)
            .on_conflict_do_update(index_elements=["tournament_key"], set_=values)
        )
    return keys


def finish_missing_tournaments(conn: Connection, seen: set[str]) -> None:
    """A tournament no longer listed as running or upcoming is over."""
    conn.execute(
        update(schedule_tournaments)
        .where(
            schedule_tournaments.c.status.in_(["running", "upcoming"]),
            schedule_tournaments.c.tournament_key.not_in(seen),
        )
        .values(status="finished", updated_at_utc=datetime.now(UTC))
    )


def upsert_matches(conn: Connection, rows: list[dict[str, Any]], index: TeamIndex) -> list[str]:
    now = datetime.now(UTC)
    ids = []
    for m in rows:
        fid = f"ps:{m['id']}"
        ids.append(fid)
        opps = [o["opponent"] for o in m.get("opponents") or []] + [None, None]
        a, b = opps[0], opps[1]
        scores = {r["team_id"]: r["score"] for r in m.get("results") or []}
        score_a = scores.get(a["id"], 0) if a else 0
        score_b = scores.get(b["id"], 0) if b else 0
        status = STATUS.get(m.get("status"), "upcoming")
        winner = None
        if status == "finished":
            won = m.get("winner_id")
            winner = "a" if a and won == a["id"] else "b" if b and won == b["id"] else None
            if winner is None and score_a == score_b and (score_a or score_b):
                winner = "draw"
        league = " ".join(
            x for x in ((m.get("league") or {}).get("name"), (m.get("serie") or {}).get("full_name")) if x
        )
        tournament = m.get("tournament") or {}
        values = {
            "source": "pandascore",
            "tournament_key": f"ps:{m['tournament_id']}" if m.get("tournament_id") else None,
            "league_name": league or None,
            "tier": tournament.get("tier"),
            "begin_at_utc": _ts(m.get("begin_at") or m.get("scheduled_at")),
            "best_of": m.get("number_of_games") or 3,
            "team_a_id": link_team(conn, a, index) if a else None,
            "team_b_id": link_team(conn, b, index) if b else None,
            "team_a_name": a["name"].strip() if a and a.get("name") else None,
            "team_b_name": b["name"].strip() if b and b.get("name") else None,
            "ps_team_a_id": a["id"] if a else None,
            "ps_team_b_id": b["id"] if b else None,
            "status": status,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
            "updated_at_utc": now,
        }
        conn.execute(
            dialect_insert(conn, fixtures)
            .values(fixture_id=fid, **values)
            .on_conflict_do_update(index_elements=["fixture_id"], set_=values)
        )
    return ids


def sync_schedule(engine: Engine, client: Any | None = None) -> dict[str, int]:
    """Tournaments (running + upcoming), upcoming matches, and matches finished in the last few days."""
    own = client is None
    client = client or make_client()
    now = datetime.now(UTC)
    window = f"{now - PAST_WINDOW:%Y-%m-%dT%H:%M:%SZ},{now:%Y-%m-%dT%H:%M:%SZ}"
    try:
        t_running = fetch_all(client, "/dota2/tournaments/running")
        t_upcoming = fetch_all(client, "/dota2/tournaments/upcoming")
        upcoming = fetch_all(client, "/dota2/matches/upcoming", sort="begin_at")
        past = fetch_all(client, "/dota2/matches/past", sort="-begin_at", **{"range[begin_at]": window})
    finally:
        if own:
            client.close()
    with engine.begin() as conn:
        for endpoint, payload in (
            ("/dota2/tournaments/running", t_running),
            ("/dota2/tournaments/upcoming", t_upcoming),
            ("/dota2/matches/upcoming", upcoming),
            ("/dota2/matches/past", past),
        ):
            save_raw_payload(conn, PROVIDER, endpoint, request_key="all", payload=payload, fetched_at_utc=now)
        seen = upsert_tournaments(conn, t_running, "running") | upsert_tournaments(conn, t_upcoming, "upcoming")
        finish_missing_tournaments(conn, seen)
        index = TeamIndex.from_conn(conn)
        n = len(upsert_matches(conn, upcoming, index)) + len(upsert_matches(conn, past, index))
    logger.info("PandaScore schedule synced", tournaments=len(seen), matches=n)
    return {"tournaments": len(seen), "matches": n}


def sync_running(engine: Engine, client: Any | None = None) -> int:
    """Live series scores. Called often, so payloads are not archived."""
    own = client is None
    client = client or make_client()
    try:
        running = fetch_all(client, "/dota2/matches/running")
    finally:
        if own:
            client.close()
    with engine.begin() as conn:
        upsert_matches(conn, running, TeamIndex.from_conn(conn))
    return len(running)
