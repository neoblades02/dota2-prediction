"""Multi-step pipelines shared by the CLI and the server: normalize everything, and a first-time fill."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Connection, Engine

from dota_predictor.ingest.explorer import (
    sync_explorer_match_index,
    sync_explorer_player_matches,
    sync_explorer_teams,
)
from dota_predictor.ingest.opendota import sync_reference_data
from dota_predictor.ingest.valve_patches import sync_valve_patches
from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.entities import (
    normalize_heroes_payload,
    normalize_leagues_payload,
    normalize_pro_players_payload,
)
from dota_predictor.normalize.games import normalize_all_pending_matches, normalize_index_matches_to_games
from dota_predictor.normalize.patches import normalize_dotaconstants_patches, normalize_valve_patches
from dota_predictor.storage.models import games, raw_ingest_objects
from dota_predictor.storage.raw_store import load_raw_payload

logger = get_logger("pipeline")


def _latest_payload(conn: Connection, endpoint: str) -> Any:
    row = (
        conn.execute(
            raw_ingest_objects.select()
            .where(raw_ingest_objects.c.endpoint == endpoint)
            .order_by(raw_ingest_objects.c.fetched_at_utc.desc())
        )
        .mappings()
        .first()
    )
    return load_raw_payload(row["payload_path"]) if row else None


def normalize_reference(engine: Engine) -> dict[str, int]:
    """Patches, heroes, leagues and pro players from the latest stored payloads."""
    counts: dict[str, int] = {}
    steps: list[tuple[str, str, Callable[[Connection, list], int]]] = [
        ("dotaconstants_patches", "/dotaconstants/patch", normalize_dotaconstants_patches),
        ("heroes", "/heroStats", normalize_heroes_payload),
        ("tournaments", "/leagues", normalize_leagues_payload),
        ("pro_players", "/proPlayers", normalize_pro_players_payload),
    ]
    with engine.begin() as conn:
        for key, endpoint, fn in steps:
            payload = _latest_payload(conn, endpoint)
            if isinstance(payload, list):
                counts[key] = fn(conn, payload)
        counts["valve_patches"] = normalize_valve_patches(conn)
    return counts


def normalize_all(engine: Engine) -> dict[str, int]:
    """Reference data, then pending match details and index-only games."""
    counts = normalize_reference(engine)
    counts["detailed_matches"] = normalize_all_pending_matches(engine)
    counts["index_games"] = normalize_index_matches_to_games(engine)
    return counts


def refresh_reference(engine: Engine) -> dict[str, Any]:
    """Daily: new leagues, heroes, pro player names and patches (no match normalization)."""
    sync_reference_data(engine)
    sync_valve_patches(engine)
    return normalize_reference(engine)


def has_games(engine: Engine) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(select(func.count()).select_from(games)).scalar())


def bootstrap(engine: Engine, since_utc: datetime = datetime(2022, 1, 1, tzinfo=UTC)) -> dict[str, Any]:
    """First-time fill of an empty database from OpenDota (~110 explorer requests, roughly 40 minutes).

    Reference data and patches are normalized before matches so every game gets its patch.
    """
    out: dict[str, Any] = {}
    steps: list[tuple[str, Callable[[], Any]]] = [
        ("reference", lambda: sync_reference_data(engine)),
        ("patches", lambda: sync_valve_patches(engine)),
        ("normalize_reference", lambda: normalize_all(engine)),
        ("explorer_index", lambda: sync_explorer_match_index(engine, since_utc=since_utc)),
        ("explorer_players", lambda: sync_explorer_player_matches(engine)),
        ("explorer_teams", lambda: sync_explorer_teams(engine)),
        ("normalize", lambda: normalize_all(engine)),
    ]
    for name, step in steps:
        logger.info("Bootstrap step starting", step=name)
        out[name] = step()
        logger.info("Bootstrap step done", step=name, result=out[name])
    return out
