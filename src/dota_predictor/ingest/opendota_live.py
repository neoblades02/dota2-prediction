"""OpenDota /live: league games in progress, with hero picks as the draft happens.

Polled every couple of minutes, so payloads are not archived; what matters is
recorded in fixtures and the prediction log.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dota_predictor.ingest.base import RateLimitedClient

BASE_URL = "https://api.opendota.com/api"


def make_client() -> RateLimitedClient:
    return RateLimitedClient(provider="opendota", base_url=BASE_URL)


def fetch_live(client: Any) -> list[dict[str, Any]]:
    resp = client.get("/live")
    resp.raise_for_status()
    return resp.json() or []


def parse_live(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """League games between two known teams. Sides are lists of (account_id, hero_id); hero 0 = not picked."""
    out = []
    for g in payload:
        if not (g.get("league_id") and g.get("team_id_radiant") and g.get("team_id_dire")):
            continue
        if g.get("deactivate_time"):  # the feed keeps ended games around for a while
            continue
        players = g.get("players") or []
        out.append(
            {
                "match_id": int(g["match_id"]) if g.get("match_id") else None,
                "series_id": g.get("series_id") or None,
                "league_id": g["league_id"],
                "radiant_team_id": g["team_id_radiant"],
                "dire_team_id": g["team_id_dire"],
                "radiant_team_name": g.get("team_name_radiant"),
                "dire_team_name": g.get("team_name_dire"),
                "radiant": [(p.get("account_id"), p.get("hero_id") or 0) for p in players if p.get("team") == 0],
                "dire": [(p.get("account_id"), p.get("hero_id") or 0) for p in players if p.get("team") == 1],
                "game_time": g.get("game_time"),
                "radiant_lead": g.get("radiant_lead"),
                "started_at": datetime.fromtimestamp(g["activate_time"], tz=UTC) if g.get("activate_time") else None,
            }
        )
    return out
