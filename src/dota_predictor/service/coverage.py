"""Cross-check the schedule against what OpenDota actually saw, so a missed major shows up as an alert.

Leagues are matched through games, not names: an OpenDota league counts as covered
when one of its recent games has a PandaScore fixture with the same two teams
within a day. League names differ too much between providers to trust.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.engine import Connection

from dota_predictor.features.asof import aware_utc
from dota_predictor.storage.models import fixtures, pro_match_index, schedule_tournaments, tournaments

ACTIVE_WINDOW = timedelta(days=3)
MATCH_SLACK = timedelta(days=1)
MAJOR_TIERS = ("s", "a")


def coverage_alerts(conn: Connection, now: datetime, schedule_enabled: bool) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if not schedule_enabled:
        msg = "No PandaScore key: matches appear only once their lobby is live."
        return [{"kind": "schedule_disabled", "message": msg}]

    ps = conn.execute(
        select(fixtures.c.team_a_id, fixtures.c.team_b_id, fixtures.c.begin_at_utc).where(
            fixtures.c.source == "pandascore", fixtures.c.team_a_id.is_not(None), fixtures.c.team_b_id.is_not(None)
        )
    ).all()
    scheduled: dict[frozenset[int], list[datetime]] = {}
    for a, b, begin in ps:
        if begin is not None:
            scheduled.setdefault(frozenset((a, b)), []).append(aware_utc(begin))

    # Only premium (major) OpenDota leagues: lower-tier qualifiers are full of teams the schedule can't link.
    recent = conn.execute(
        select(
            pro_match_index.c.leagueid,
            pro_match_index.c.league_name,
            pro_match_index.c.radiant_team_id,
            pro_match_index.c.dire_team_id,
            pro_match_index.c.start_time_utc,
        )
        .join(tournaments, tournaments.c.leagueid == pro_match_index.c.leagueid)
        .where(pro_match_index.c.start_time_utc >= now - ACTIVE_WINDOW, tournaments.c.tier == "premium")
    ).all()
    leagues: dict[int, dict[str, Any]] = {}
    for league_id, name, r, d, start in recent:
        info = leagues.setdefault(league_id, {"name": name, "games": 0, "covered": False, "last": None})
        start = aware_utc(start)
        info["games"] += 1
        info["last"] = max(info["last"] or start, start)
        if any(abs(t - start) <= MATCH_SLACK for t in scheduled.get(frozenset((r, d)), [])):
            info["covered"] = True
    for league_id, info in sorted(leagues.items(), key=lambda kv: -kv[1]["games"]):
        if not info["covered"]:
            alerts.append(
                {
                    "kind": "missing_from_schedule",
                    "leagueid": league_id,
                    "message": f"{info['name'] or league_id}: {info['games']} games on OpenDota in the last 3 days, "
                    "none matched to the schedule. Live games from it are still picked up from OpenDota.",
                    "last_game": info["last"],
                }
            )

    upcoming_or_live = (
        select(fixtures.c.tournament_key, func.count().label("n"))
        .where(
            or_(
                fixtures.c.status == "running",
                (fixtures.c.status == "upcoming") & (fixtures.c.begin_at_utc <= now + timedelta(hours=24)),
            )
        )
        .group_by(fixtures.c.tournament_key)
    )
    active = dict(conn.execute(upcoming_or_live).all())
    for t in conn.execute(
        select(schedule_tournaments).where(
            schedule_tournaments.c.status == "running", schedule_tournaments.c.tier.in_(MAJOR_TIERS)
        )
    ).mappings():
        if not active.get(t["tournament_key"]):
            title = " ".join(x for x in (t["league_name"], t["serie_name"], t["stage_name"]) if x)
            msg = f"{title} (tier {t['tier'].upper()}) is running but lists no match in the next 24 hours."
            alerts.append({"kind": "no_matches_listed", "message": msg})
    return alerts
