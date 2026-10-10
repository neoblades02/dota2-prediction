import pytest
from sqlalchemy import create_engine

from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.quality.checks import run_data_quality_audit
from dota_predictor.quality.coverage import generate_coverage_report
from dota_predictor.storage.repository import init_db


@pytest.fixture
def test_engine(tmp_path):
    db_file = tmp_path / "test_quality.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)
    return engine


def test_quality_audit_and_report(test_engine, tmp_path):
    sample_match = {
        "match_id": 999901,
        "start_time": 1726000000,
        "duration": 2000,
        "radiant_win": True,
        "radiant_team_id": 1,
        "dire_team_id": 2,
        "patch": 60,
        "players": [{"account_id": i, "hero_id": i % 10 + 1, "player_slot": i, "isRadiant": i < 5} for i in range(10)],
    }

    with test_engine.begin() as conn:
        normalize_match_detail(conn, sample_match)
        audit = run_data_quality_audit(conn)

    assert audit["counts"]["normalized_games"] == 1
    assert audit["counts"]["participants"] == 10
    assert audit["quality_metrics"]["duplicate_match_ids"] == 0
    assert audit["quality_metrics"]["games_with_non_10_participants"] == 0
    assert audit["quality_metrics"]["games_with_unknown_results"] == 0

    generate_coverage_report(test_engine, output_dir=tmp_path / "reports")
    assert (tmp_path / "reports" / "coverage.json").exists()
    assert (tmp_path / "reports" / "coverage.md").exists()
