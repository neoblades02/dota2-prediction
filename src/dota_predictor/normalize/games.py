from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection, Engine

from dota_predictor.logging_config import get_logger
from dota_predictor.normalize.entities import (
    get_or_create_player,
    get_or_create_team,
)
from dota_predictor.normalize.patches import resolve_patch_for_match
from dota_predictor.storage.models import (
    draft_actions,
    game_participants,
    games,
    lineup_snapshots,
    raw_ingest_objects,
    tournaments,
)
from dota_predictor.storage.raw_store import load_raw_payload
from dota_predictor.storage.upsert import dialect_insert

logger = get_logger("normalize.games")


def compute_lineup_hash(player_ids: list[int | None]) -> tuple[str, str]:
    """Compute sorted JSON string and SHA256 hash of a 5-player lineup."""
    cleaned = sorted([pid for pid in player_ids if pid is not None])
    json_str = json.dumps(cleaned)
    hash_str = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    return json_str, hash_str


def normalize_match_detail(
    conn: Connection,
    payload: dict[str, Any],
    raw_object_id: int | None = None,
) -> int | None:
    """Normalize a single match detail payload into canonical games, participants, draft, and lineups."""
    match_id = payload.get("match_id")
    if not match_id:
        return None

    now = datetime.now(UTC)
    start_epoch = payload.get("start_time")
    start_utc = datetime.fromtimestamp(start_epoch, tz=UTC) if start_epoch else None
    duration_s = payload.get("duration")
    radiant_win = payload.get("radiant_win")

    # 1. Resolve Teams
    radiant_team_id_raw = payload.get("radiant_team_id")
    dire_team_id_raw = payload.get("dire_team_id")

    radiant_team_id = (
        get_or_create_team(
            conn=conn,
            provider_team_id=radiant_team_id_raw,
            name=payload.get("radiant_name"),
            observed_at_utc=start_utc,
        )
        if radiant_team_id_raw
        else None
    )

    dire_team_id = (
        get_or_create_team(
            conn=conn,
            provider_team_id=dire_team_id_raw,
            name=payload.get("dire_name"),
            observed_at_utc=start_utc,
        )
        if dire_team_id_raw
        else None
    )

    # 2. Resolve Tournament
    leagueid = payload.get("leagueid")
    tournament_id = None
    if leagueid:
        t_row = conn.execute(
            select(tournaments.c.tournament_id).where(tournaments.c.leagueid == leagueid)
        ).scalar_one_or_none()
        if t_row is None:
            # Insert league
            l_info = payload.get("league") or {}
            res_t = conn.execute(
                dialect_insert(conn, tournaments).values(
                    leagueid=leagueid,
                    title=l_info.get("name") or f"League {leagueid}",
                    tier=l_info.get("tier"),
                    metadata_source="opendota",
                )
            )
            tournament_id = res_t.inserted_primary_key[0]
        else:
            tournament_id = t_row

    # 3. Resolve Patch
    raw_patch = payload.get("patch")
    patch_id, patch_conf = resolve_patch_for_match(conn=conn, match_time_utc=start_utc, raw_patch_id=raw_patch)

    # 4. Upsert canonical game
    stmt_game = (
        dialect_insert(conn, games)
        .values(
            valve_match_id=match_id,
            tournament_id=tournament_id,
            radiant_team_id=radiant_team_id,
            dire_team_id=dire_team_id,
            start_utc=start_utc,
            duration_s=duration_s,
            radiant_win=radiant_win,
            patch_id=patch_id,
            patch_confidence=patch_conf,
            parse_status="parsed" if payload.get("players") else "unparsed",
            source="opendota",
            raw_object_id=raw_object_id,
            ingested_at_utc=now,
        )
        .on_conflict_do_update(
            index_elements=["valve_match_id"],
            set_={
                "duration_s": duration_s,
                "radiant_win": radiant_win,
                "patch_id": patch_id,
                "patch_confidence": patch_conf,
                "parse_status": "parsed" if payload.get("players") else "unparsed",
                "tournament_id": tournament_id,
                "radiant_team_id": radiant_team_id,
                "dire_team_id": dire_team_id,
            },
        )
    )
    conn.execute(stmt_game)

    game_id = conn.execute(select(games.c.game_id).where(games.c.valve_match_id == match_id)).scalar_one()

    # 5. Participants & Lineup reconstruction
    players_data = payload.get("players") or []
    radiant_pids: list[int | None] = []
    dire_pids: list[int | None] = []

    for p in players_data:
        account_id = p.get("account_id")
        slot = p.get("player_slot")
        is_radiant = p.get("isRadiant")
        if is_radiant is None and slot is not None:
            is_radiant = slot < 128
        side = "radiant" if is_radiant else "dire"
        team_id = radiant_team_id if is_radiant else dire_team_id

        player_id = get_or_create_player(
            conn=conn,
            account_id=account_id,
            name=p.get("name"),
            personaname=p.get("personaname"),
            observed_at_utc=start_utc,
        )

        if is_radiant:
            radiant_pids.append(player_id)
        else:
            dire_pids.append(player_id)

        hero_id = p.get("hero_id")

        if slot is not None:
            stmt_part = (
                dialect_insert(conn, game_participants)
                .values(
                    game_id=game_id,
                    player_id=player_id,
                    account_id=account_id,
                    hero_id=hero_id,
                    side=side,
                    team_id=team_id,
                    player_slot=slot,
                    role_estimate=None,
                    role_confidence="unknown",
                    standin_flag=False,
                    name_snapshot=p.get("name"),
                    personaname_snapshot=p.get("personaname"),
                )
                .on_conflict_do_update(
                    index_elements=["game_id", "player_slot"],
                    set_={
                        "player_id": player_id,
                        "hero_id": hero_id,
                        "side": side,
                        "team_id": team_id,
                        "account_id": account_id,
                    },
                )
            )
            conn.execute(stmt_part)

    # 6. Team Lineups ("their team composition")
    if radiant_team_id and radiant_pids:
        r_json, r_hash = compute_lineup_hash(radiant_pids)
        conn.execute(
            dialect_insert(conn, lineup_snapshots)
            .values(
                game_id=game_id,
                team_id=radiant_team_id,
                player_ids_json=r_json,
                player_set_hash=r_hash,
                confidence="confirmed" if len([p for p in radiant_pids if p]) >= 5 else "partial",
            )
            .on_conflict_do_update(
                index_elements=["game_id", "team_id"],
                set_={
                    "player_ids_json": r_json,
                    "player_set_hash": r_hash,
                },
            )
        )

    if dire_team_id and dire_pids:
        d_json, d_hash = compute_lineup_hash(dire_pids)
        conn.execute(
            dialect_insert(conn, lineup_snapshots)
            .values(
                game_id=game_id,
                team_id=dire_team_id,
                player_ids_json=d_json,
                player_set_hash=d_hash,
                confidence="confirmed" if len([p for p in dire_pids if p]) >= 5 else "partial",
            )
            .on_conflict_do_update(
                index_elements=["game_id", "team_id"],
                set_={
                    "player_ids_json": d_json,
                    "player_set_hash": d_hash,
                },
            )
        )

    # 7. Draft Actions (picks and bans sequence)
    picks_bans = payload.get("picks_bans") or []
    for pb in picks_bans:
        order = pb.get("order")
        if order is None:
            continue
        is_pick = bool(pb.get("is_pick"))
        pb_hero_id = pb.get("hero_id")
        pb_team = pb.get("team")
        pb_side = "radiant" if pb_team == 0 else "dire"

        conn.execute(
            dialect_insert(conn, draft_actions)
            .values(
                game_id=game_id,
                order_seq=order,
                is_pick=is_pick,
                hero_id=pb_hero_id,
                side=pb_side,
                team_number_raw=pb_team,
                source="opendota",
            )
            .on_conflict_do_update(
                index_elements=["game_id", "order_seq"],
                set_={
                    "is_pick": is_pick,
                    "hero_id": pb_hero_id,
                    "side": pb_side,
                },
            )
        )

    return game_id


def normalize_all_pending_matches(engine: Engine) -> int:
    """Normalize all raw match payloads in raw_ingest_objects that haven't been normalized."""
    count = 0
    with engine.begin() as conn:
        raw_rows = (
            conn.execute(
                select(raw_ingest_objects).where(
                    raw_ingest_objects.c.provider == "opendota",
                    raw_ingest_objects.c.endpoint.like("/matches/%"),
                )
            )
            .mappings()
            .fetchall()
        )

        for r in raw_rows:
            try:
                payload = load_raw_payload(r["payload_path"])
                if isinstance(payload, dict) and "match_id" in payload:
                    gid = normalize_match_detail(conn, payload, raw_object_id=r["id"])
                    if gid:
                        count += 1
            except Exception as e:
                logger.error("Failed to normalize match payload", id=r["id"], error=str(e))

    logger.info("Normalized match details", total=count)
    return count


def normalize_index_matches_to_games(engine: Engine, limit: int | None = None) -> int:
    """Normalize indexed pro matches directly into canonical games table."""
    from dota_predictor.storage.models import pro_match_index

    count = 0
    with engine.begin() as conn:
        # Find match_ids in pro_match_index not yet in games
        existing_match_ids = set(conn.execute(select(games.c.valve_match_id)).scalars().all())

        query = (
            select(pro_match_index)
            .where(
                pro_match_index.c.radiant_win.is_not(None),
                pro_match_index.c.radiant_team_id.is_not(None),
                pro_match_index.c.dire_team_id.is_not(None),
            )
            .order_by(pro_match_index.c.start_time_utc.asc())
        )
        if limit is not None:
            query = query.limit(limit)

        rows = conn.execute(query).mappings().fetchall()
        now = datetime.now(UTC)

        for row in rows:
            mid = row["match_id"]
            if mid in existing_match_ids:
                continue

            start_utc = row["start_time_utc"]
            r_team_id = get_or_create_team(
                conn=conn,
                provider_team_id=row["radiant_team_id"],
                name=row["radiant_name"],
                observed_at_utc=start_utc,
            )
            d_team_id = get_or_create_team(
                conn=conn,
                provider_team_id=row["dire_team_id"],
                name=row["dire_name"],
                observed_at_utc=start_utc,
            )

            # Tournament
            tournament_id = None
            if row["leagueid"]:
                t_row = conn.execute(
                    select(tournaments.c.tournament_id).where(tournaments.c.leagueid == row["leagueid"])
                ).scalar_one_or_none()
                if t_row is None:
                    res_t = conn.execute(
                        dialect_insert(conn, tournaments).values(
                            leagueid=row["leagueid"],
                            title=row["league_name"] or f"League {row['leagueid']}",
                            metadata_source="opendota",
                        )
                    )
                    tournament_id = res_t.inserted_primary_key[0]
                else:
                    tournament_id = t_row

            # Patch (index rows carry no patch field; `version` is the replay parser version)
            patch_id, patch_conf = resolve_patch_for_match(conn=conn, match_time_utc=start_utc)

            stmt = (
                dialect_insert(conn, games)
                .values(
                    valve_match_id=mid,
                    tournament_id=tournament_id,
                    radiant_team_id=r_team_id,
                    dire_team_id=d_team_id,
                    start_utc=start_utc,
                    duration_s=row["duration_s"],
                    radiant_win=row["radiant_win"],
                    patch_id=patch_id,
                    patch_confidence=patch_conf,
                    parse_status="index_only",
                    source="opendota",
                    ingested_at_utc=now,
                )
                .on_conflict_do_nothing()
            )
            conn.execute(stmt)
            existing_match_ids.add(mid)
            count += 1

    logger.info("Normalized indexed matches to games", count=count)
    return count
