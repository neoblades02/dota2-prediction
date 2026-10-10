import math
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, func, insert, select

from dota_predictor.normalize.entities import normalize_heroes_payload
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.predict.live_draft import LiveDraftPredictor
from dota_predictor.predict.markets import series_markets
from dota_predictor.predict.service import PredictionService
from dota_predictor.storage.models import fixtures, prediction_results, predictions
from dota_predictor.storage.repository import init_db

NOW = datetime(2026, 10, 10, tzinfo=UTC)


def _match(i, r, d, radiant_win):
    accounts = [r * 10 + k for k in range(5)] + [d * 10 + k for k in range(5)]
    return {
        "match_id": 70000 + i,
        "start_time": 1720000000 + i * 3600,
        "duration": 2000,
        "radiant_win": radiant_win,
        "radiant_team_id": r,
        "dire_team_id": d,
        "players": [
            {"account_id": a, "hero_id": (i + k) % 20 + 1, "player_slot": k, "isRadiant": k < 5}
            for k, a in enumerate(accounts)
        ],
    }


@pytest.fixture
def service(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 's.db').as_posix()}")
    init_db(engine)
    with engine.begin() as conn:
        normalize_heroes_payload(
            conn, [{"id": h, "name": f"npc_dota_hero_h{h}", "localized_name": f"Hero{h}"} for h in range(1, 21)]
        )
        for i in range(30):  # team 1 beats team 2 most of the time
            normalize_match_detail(conn, _match(i, 1, 2, i % 4 != 0) if i % 2 else _match(i, 2, 1, i % 4 == 0))
    predictor = LiveDraftPredictor(engine=engine, model_path=tmp_path / "x.joblib", predraft_model_path=tmp_path / "y")
    return PredictionService(engine, predictor)


def _fx(**kw):
    base = {
        "fixture_id": "ps:1",
        "source": "pandascore",
        "best_of": 3,
        "status": "upcoming",
        "team_a_id": 1,
        "team_b_id": 2,
        "score_a": 0,
        "score_b": 0,
        "tier": "a",
        "updated_at_utc": NOW,
    }
    return base | kw


def test_predict_fixture_is_side_neutral_with_markets(service):
    pred = service.predict_fixture(_fx())
    expected = service.predictor.predict_side_neutral(1, 2)["p_map"]
    assert pred["p_map"] == pytest.approx(expected)
    assert pred["p_map"] > 0.5
    expected_mk = series_markets(expected, 3)
    for group in ("match_winner", "correct_score", "handicap", "map1_match"):
        assert pred["markets"][group] == pytest.approx(expected_mk[group])
    assert pred["p_series"] == pytest.approx(pred["markets"]["match_winner"]["a"])
    assert service.predict_fixture(_fx(team_b_id=None)) is None
    assert service.predict_fixture(_fx(best_of=4)) is None


def test_low_data_flag(service):
    assert service.predict_fixture(_fx())["low_data"] is False
    service.low_data_games = 31
    assert service.predict_fixture(_fx())["low_data"] is True


def test_live_draft_uses_side_orientation(service):
    live = {
        "radiant_team_id": 2,
        "dire_team_id": 1,
        "radiant": [(20 + k, k + 1) for k in range(5)],
        "dire": [(10 + k, k + 6) for k in range(5)],
    }
    out = service.predict_live_game(_fx(), live)
    raw = service.predictor.predict_live_draft(
        2,
        1,
        radiant_heroes=[1, 2, 3, 4, 5],
        dire_heroes=[6, 7, 8, 9, 10],
        radiant_players=[service.predictor.account_to_player[20 + k] for k in range(5)],
        dire_players=[service.predictor.account_to_player[10 + k] for k in range(5)],
    )
    assert out["p_map_now"] == pytest.approx(1 - raw["probabilities"]["p_radiant"])
    assert out["map_number"] == 1
    assert service.predict_live_game(_fx(), {**live, "dire": live["dire"][:4]}) is None


def test_log_is_idempotent_and_grading_scores_locked_rows(service):
    with service.engine.begin() as conn:
        conn.execute(
            insert(fixtures),
            [
                _fx(fixture_id="ps:1", status="finished", winner="a"),
                _fx(fixture_id="ps:2", status="finished", winner="b"),
                _fx(fixture_id="ps:3", status="finished", winner="draw", best_of=2),
            ],
        )
        for fid, p in (("ps:1", 0.7), ("ps:2", 0.4), ("ps:3", 0.5)):
            for _ in range(2):
                service.log(conn, _fx(fixture_id=fid), "locked", {"p_series": p, "p_map": p, "low_data": False})
        assert conn.execute(select(func.count()).select_from(predictions)).scalar() == 3
        assert service.grade(conn) == 3
        assert service.grade(conn) == 0
        statuses = sorted(r[0] for r in conn.execute(select(prediction_results.c.result_status)))
        assert statuses == ["void", "won", "won"]
        rec = service.record(conn)
    assert rec["overall"]["n"] == 2
    assert rec["overall"]["accuracy"] == pytest.approx(1.0)
    assert rec["overall"]["brier"] == pytest.approx((0.3**2 + 0.4**2) / 2)
    assert rec["overall"]["log_loss"] == pytest.approx(-(math.log(0.7) + math.log(0.6)) / 2)
    assert len(rec["recent"]) == 2


def test_unknown_lineup_falls_back_to_elo_instead_of_extrapolating(service, monkeypatch):
    import dota_predictor.predict.live_draft as ld

    predictor = service.predictor
    predictor.predraft_artifact = {"features": ["elo_diff"], "model": None}
    monkeypatch.setattr(ld, "predict_artifact", lambda artifact, X, elo: [0.01])  # an extrapolating model
    known = predictor.predict_live_draft(1, 2)["probabilities"]
    assert known["model"] == "lightgbm" and known["p_radiant"] == 0.01
    unknown = predictor.predict_live_draft(99, 2)["probabilities"]  # team 99 has no games and no players
    assert unknown["model"] == "elo_unknown_lineup"
    assert unknown["p_radiant"] == pytest.approx(unknown["pre_draft_radiant_win_pct"] / 100, abs=0.001)
    pred = service.predict_fixture(_fx(team_a_id=99))
    assert pred["low_data"] is True and 0.2 < pred["p_map"] < 0.8


def test_record_scores_bookmaker_and_draft_maps(service):
    from dota_predictor.storage.models import fixture_odds

    with service.engine.begin() as conn:
        conn.execute(insert(fixtures), [_fx(fixture_id="ps:1", status="finished", winner="a")])
        conn.execute(insert(fixture_odds).values(fixture_id="ps:1", odds_a=1.5, odds_b=2.5, updated_at_utc=NOW))
        service.log(conn, _fx(), "locked", {"p_series": 0.7, "p_map": 0.6})
        # map 1 of the series was OpenDota match 70001: radiant (team 1 = team A) won
        draft = {"p_series": 0.75, "p_map": 0.6, "p_map_now": 0.8, "a_side": "radiant", "match_id": "70001"}
        service.log(conn, _fx(), "live_draft", draft, 1)
        service.log(conn, _fx(), "live_draft", draft | {"match_id": 999999}, 2)  # result not known yet
        assert service.grade(conn) == 2  # the locked series row and map 1
        rec = service.record(conn)
    assert rec["book"]["n"] == 1
    assert rec["book"]["model"]["brier"] == pytest.approx(0.3**2)
    assert rec["book"]["book"]["brier"] == pytest.approx((1 - 0.625) ** 2)  # (1/1.5) / (1/1.5 + 1/2.5)
    assert rec["draft"]["n"] == 1
    assert rec["draft"]["pre_draft"]["brier"] == pytest.approx(0.4**2)
    assert rec["draft"]["post_draft"]["brier"] == pytest.approx(0.2**2)
