import pytest
from sqlalchemy import create_engine

from dota_predictor.features.asof import build_pregame_features
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.storage.repository import init_db


@pytest.fixture
def test_engine(tmp_path):
    db_file = tmp_path / "test_leakage.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)
    return engine


def test_asof_temporal_leakage(test_engine, tmp_path):
    # Match 1: at T = 1000
    m1 = {
        "match_id": 10001,
        "start_time": 1720001000,
        "duration": 2000,
        "radiant_win": True,
        "radiant_team_id": 1,
        "dire_team_id": 2,
        "patch": 60,
        "players": [{"account_id": i + 1, "hero_id": i + 1, "player_slot": i, "isRadiant": i < 5} for i in range(10)],
    }

    # Match 2: at T = 2000
    m2 = {
        "match_id": 10002,
        "start_time": 1720002000,
        "duration": 2200,
        "radiant_win": False,
        "radiant_team_id": 1,
        "dire_team_id": 2,
        "patch": 60,
        "players": [
            {"account_id": i + 1, "hero_id": (i + 5) % 10 + 1, "player_slot": i, "isRadiant": i < 5} for i in range(10)
        ],
    }

    with test_engine.begin() as conn:
        normalize_match_detail(conn, m1)
        normalize_match_detail(conn, m2)

    features_path_1 = tmp_path / "feat1.parquet"
    df1 = build_pregame_features(test_engine, output_path=features_path_1)
    assert len(df1) == 2

    # Snapshot features for Match 2
    m2_features_before = df1.filter(df1["valve_match_id"] == 10002).to_dicts()[0]

    # Ingest Match 3: at T = 3000 (IN THE FUTURE)
    m3 = {
        "match_id": 10003,
        "start_time": 1720003000,
        "duration": 1800,
        "radiant_win": True,
        "radiant_team_id": 1,
        "dire_team_id": 2,
        "patch": 60,
        "players": [{"account_id": i + 1, "hero_id": i + 2, "player_slot": i, "isRadiant": i < 5} for i in range(10)],
    }
    with test_engine.begin() as conn:
        normalize_match_detail(conn, m3)

    # Rebuild features
    features_path_2 = tmp_path / "feat2.parquet"
    df2 = build_pregame_features(test_engine, output_path=features_path_2)
    assert len(df2) == 3

    m2_features_after = df2.filter(df2["valve_match_id"] == 10002).to_dicts()[0]

    # LEAKAGE CHECK: Features for Match 2 must be EXACTLY equal before and after Match 3 was added
    assert m2_features_before == m2_features_after
    assert m2_features_before["lineup_5man_shared_radiant"] == 1
    assert m2_features_before["target_radiant_win"] == 0
