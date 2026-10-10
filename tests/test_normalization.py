import json

import pytest
from sqlalchemy import create_engine, select

from dota_predictor.normalize.entities import (
    get_or_create_player,
    get_or_create_team,
    normalize_heroes_payload,
    normalize_leagues_payload,
)
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.storage.models import (
    draft_actions,
    game_participants,
    games,
    lineup_snapshots,
)
from dota_predictor.storage.repository import init_db


@pytest.fixture
def test_engine(tmp_path):
    db_file = tmp_path / "test_norm.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)
    return engine


def test_entity_normalization(test_engine):
    with test_engine.begin() as conn:
        t_id = get_or_create_team(conn, 111, name="Team Aster")
        assert t_id == 111

        p_id = get_or_create_player(conn, 99999, name="Somnus", personaname="Maybe")
        assert p_id is not None

        # Verify hero normalization
        heroes = [
            {"id": 1, "name": "npc_dota_hero_antimage", "localized_name": "Anti-Mage"},
            {"id": 2, "name": "npc_dota_hero_axe", "localized_name": "Axe"},
        ]
        n_heroes = normalize_heroes_payload(conn, heroes)
        assert n_heroes == 2

        # Verify leagues normalization
        leagues = [{"leagueid": 100, "name": "The International", "tier": "premium"}]
        n_leagues = normalize_leagues_payload(conn, leagues)
        assert n_leagues == 1


def test_match_detail_normalization_and_idempotency(test_engine):
    # Construct a sample 10-player match detail
    sample_match = {
        "match_id": 8000001,
        "start_time": 1725000000,
        "duration": 2400,
        "radiant_win": True,
        "radiant_team_id": 10,
        "radiant_name": "Team Secret",
        "dire_team_id": 20,
        "dire_name": "OG",
        "leagueid": 500,
        "patch": 60,
        "players": [
            # 5 Radiant players (slots 0-4)
            {"account_id": 101, "hero_id": 1, "player_slot": 0, "isRadiant": True, "name": "Nisha"},
            {"account_id": 102, "hero_id": 2, "player_slot": 1, "isRadiant": True, "name": "MidOne"},
            {"account_id": 103, "hero_id": 3, "player_slot": 2, "isRadiant": True, "name": "Zai"},
            {"account_id": 104, "hero_id": 4, "player_slot": 3, "isRadiant": True, "name": "YapzOr"},
            {"account_id": 105, "hero_id": 5, "player_slot": 4, "isRadiant": True, "name": "Puppey"},
            # 5 Dire players (slots 128-132)
            {"account_id": 201, "hero_id": 6, "player_slot": 128, "isRadiant": False, "name": "Ana"},
            {"account_id": 202, "hero_id": 7, "player_slot": 129, "isRadiant": False, "name": "Topson"},
            {"account_id": 203, "hero_id": 8, "player_slot": 130, "isRadiant": False, "name": "Ceb"},
            {"account_id": 204, "hero_id": 9, "player_slot": 131, "isRadiant": False, "name": "JerAx"},
            {"account_id": 205, "hero_id": 10, "player_slot": 132, "isRadiant": False, "name": "N0tail"},
        ],
        "picks_bans": [
            {"order": 0, "is_pick": False, "hero_id": 50, "team": 0},
            {"order": 1, "is_pick": False, "hero_id": 51, "team": 1},
            {"order": 2, "is_pick": True, "hero_id": 1, "team": 0},
        ],
    }

    with test_engine.begin() as conn:
        gid = normalize_match_detail(conn, sample_match)
        assert gid is not None

        # Check games
        game = conn.execute(select(games).where(games.c.game_id == gid)).mappings().first()
        assert game["valve_match_id"] == 8000001
        assert game["radiant_win"] is True
        assert game["radiant_team_id"] == 10
        assert game["dire_team_id"] == 20

        # Check participants (exactly 10)
        parts = conn.execute(select(game_participants).where(game_participants.c.game_id == gid)).mappings().fetchall()
        assert len(parts) == 10

        radiant_parts = [p for p in parts if p["side"] == "radiant"]
        dire_parts = [p for p in parts if p["side"] == "dire"]
        assert len(radiant_parts) == 5
        assert len(dire_parts) == 5

        # Check lineup snapshots
        lineups = conn.execute(select(lineup_snapshots).where(lineup_snapshots.c.game_id == gid)).mappings().fetchall()
        assert len(lineups) == 2
        for lineup in lineups:
            pids = json.loads(lineup["player_ids_json"])
            assert len(pids) == 5

        # Check draft actions
        drafts = conn.execute(select(draft_actions).where(draft_actions.c.game_id == gid)).mappings().fetchall()
        assert len(drafts) == 3

        # Test idempotency: re-running does not duplicate rows
        gid2 = normalize_match_detail(conn, sample_match)
        assert gid == gid2

        parts2 = conn.execute(select(game_participants).where(game_participants.c.game_id == gid)).fetchall()
        assert len(parts2) == 10
