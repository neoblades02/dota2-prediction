from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.engine import Connection

from dota_predictor.storage.models import (
    draft_actions,
    game_participants,
    games,
    lineup_snapshots,
    patches,
    players,
    pro_match_index,
    teams,
    tournaments,
)


def run_data_quality_audit(conn: Connection) -> dict[str, Any]:
    """Audit data quality and integrity across canonical entities and matches."""
    # 1. Total counts
    total_indexed = conn.execute(select(func.count(pro_match_index.c.match_id))).scalar() or 0
    total_games = conn.execute(select(func.count(games.c.game_id))).scalar() or 0
    total_participants = conn.execute(select(func.count(game_participants.c.id))).scalar() or 0
    total_players = conn.execute(select(func.count(players.c.player_id))).scalar() or 0
    total_teams = conn.execute(select(func.count(teams.c.team_id))).scalar() or 0
    total_lineups = conn.execute(select(func.count(lineup_snapshots.c.id))).scalar() or 0
    total_patches = conn.execute(select(func.count(patches.c.patch_id))).scalar() or 0
    total_tournaments = conn.execute(select(func.count(tournaments.c.tournament_id))).scalar() or 0

    # 2. Duplicate valve match IDs
    dup_matches = conn.execute(
        select(games.c.valve_match_id, func.count(games.c.game_id).label("cnt"))
        .group_by(games.c.valve_match_id)
        .having(func.count(games.c.game_id) > 1)
    ).fetchall()
    dup_match_count = len(dup_matches)

    # 3. Incomplete games (participants != 10)
    part_per_game = conn.execute(
        select(game_participants.c.game_id, func.count(game_participants.c.id).label("cnt"))
        .group_by(game_participants.c.game_id)
        .having(func.count(game_participants.c.id) != 10)
    ).fetchall()
    incomplete_games_count = len(part_per_game)

    # 4. Unknown game results
    unknown_results = (
        conn.execute(select(func.count(games.c.game_id)).where(games.c.radiant_win.is_(None))).scalar() or 0
    )

    # 5. Missing team identity
    missing_teams = (
        conn.execute(
            select(func.count(games.c.game_id)).where(
                (games.c.radiant_team_id.is_(None)) | (games.c.dire_team_id.is_(None))
            )
        ).scalar()
        or 0
    )

    # 6. Missing player identity (account_id is 0 or null)
    anonymous_participants = (
        conn.execute(
            select(func.count(game_participants.c.id)).where(
                (game_participants.c.player_id.is_(None)) | (game_participants.c.account_id.is_(None))
            )
        ).scalar()
        or 0
    )

    # 7. Patch confidence breakdown
    patch_conf_rows = conn.execute(
        select(games.c.patch_confidence, func.count(games.c.game_id)).group_by(games.c.patch_confidence)
    ).fetchall()
    patch_breakdown = {row[0] or "null": row[1] for row in patch_conf_rows}

    # 8. Draft actions coverage (games with picks/bans)
    games_with_draft = conn.execute(select(func.count(func.distinct(draft_actions.c.game_id)))).scalar() or 0
    games_with_lineups = conn.execute(select(func.count(func.distinct(game_participants.c.game_id)))).scalar() or 0

    return {
        "counts": {
            "indexed_matches": total_indexed,
            "normalized_games": total_games,
            "participants": total_participants,
            "canonical_players": total_players,
            "canonical_teams": total_teams,
            "lineup_snapshots": total_lineups,
            "patches": total_patches,
            "tournaments": total_tournaments,
        },
        "quality_metrics": {
            "duplicate_match_ids": dup_match_count,
            "games_with_non_10_participants": incomplete_games_count,
            "games_with_unknown_results": unknown_results,
            "games_with_missing_teams": missing_teams,
            "anonymous_participant_slots": anonymous_participants,
            "games_with_draft_data": games_with_draft,
            "games_with_lineups": games_with_lineups,
        },
        "patch_confidence": patch_breakdown,
    }
