import pytest
from sqlalchemy import create_engine

from dota_predictor.normalize.entities import get_or_create_team, normalize_heroes_payload
from dota_predictor.predict.live_draft import LiveDraftPredictor
from dota_predictor.storage.repository import init_db


@pytest.fixture
def seeded_engine(tmp_path):
    db_file = tmp_path / "live_draft.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)

    with engine.begin() as conn:
        # Seed teams
        get_or_create_team(conn, 1001, name="Team Spirit", tag="TS")
        get_or_create_team(conn, 1002, name="Team Yandex", tag="TY")

        # Seed heroes
        heroes = [
            {"id": 1, "name": "npc_dota_hero_antimage", "localized_name": "Anti-Mage"},
            {"id": 2, "name": "npc_dota_hero_axe", "localized_name": "Axe"},
            {"id": 3, "name": "npc_dota_hero_bane", "localized_name": "Bane"},
            {"id": 4, "name": "npc_dota_hero_bloodseeker", "localized_name": "Bloodseeker"},
            {"id": 5, "name": "npc_dota_hero_crystal_maiden", "localized_name": "Crystal Maiden"},
            {"id": 6, "name": "npc_dota_hero_drow_ranger", "localized_name": "Drow Ranger"},
            {"id": 7, "name": "npc_dota_hero_earthshaker", "localized_name": "Earthshaker"},
            {"id": 8, "name": "npc_dota_hero_juggernaut", "localized_name": "Juggernaut"},
            {"id": 9, "name": "npc_dota_hero_mirana", "localized_name": "Mirana"},
            {"id": 10, "name": "npc_dota_hero_morphling", "localized_name": "Morphling"},
        ]
        normalize_heroes_payload(conn, heroes)

    return engine


def test_live_draft_predictor_resolution_and_forecast(seeded_engine, tmp_path):
    predictor = LiveDraftPredictor(engine=seeded_engine, model_path=tmp_path / "missing.joblib")

    # Resolve hero by name and fuzzy matching
    hid_am = predictor.resolve_hero("Anti-Mage")
    assert hid_am == 1
    hid_axe = predictor.resolve_hero("axe")
    assert hid_axe == 2

    # Resolve team by name and tag
    tid_ts = predictor.resolve_team("Team Spirit")
    assert tid_ts == 1001
    tid_ty = predictor.resolve_team("TY")
    assert tid_ty == 1002

    # Predict ongoing draft
    res = predictor.predict_live_draft(
        radiant_team="Team Spirit",
        dire_team="Team Yandex",
        radiant_heroes=["Anti-Mage", "Axe", "Bane", "Bloodseeker", "Crystal Maiden"],
        dire_heroes=["Drow Ranger", "Earthshaker", "Juggernaut", "Mirana", "Morphling"],
        patch_id=60,
    )

    assert "matchup" in res
    assert "draft" in res
    assert "probabilities" in res
    assert res["matchup"]["radiant_team"] == "Team Spirit"
    assert res["matchup"]["dire_team"] == "Team Yandex"
    assert len(res["draft"]["radiant_heroes"]) == 5
    assert len(res["draft"]["dire_heroes"]) == 5
    assert 0.0 <= res["probabilities"]["post_draft_radiant_win_pct"] <= 100.0
    assert 0.0 <= res["probabilities"]["post_draft_dire_win_pct"] <= 100.0
    # No games stored and no model: pure cold-start Elo
    assert res["probabilities"]["model"] == "elo_only"
    assert res["probabilities"]["pre_draft_radiant_win_pct"] == 50.0
    assert res["matchup"]["radiant_games_known"] == 0
