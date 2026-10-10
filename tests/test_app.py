import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, insert

from dota_predictor.normalize.entities import normalize_heroes_payload
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.predict.live_draft import LiveDraftPredictor
from dota_predictor.predict.service import PredictionService
from dota_predictor.service.app import create_app
from dota_predictor.service.jobs import JobRunner
from dota_predictor.storage.models import fixtures, schedule_tournaments, team_links
from dota_predictor.storage.repository import init_db

NOW = datetime.now(UTC)
AUTH = ("admin", "pw")


def _match(i, r, d, radiant_win):
    accounts = [r * 10 + k for k in range(5)] + [d * 10 + k for k in range(5)]
    return {
        "match_id": 90000 + i,
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


def _fx(fid, **kw):
    return {
        "fixture_id": fid,
        "source": "pandascore",
        "tournament_key": "ps:1",
        "best_of": 3,
        "status": "upcoming",
        "begin_at_utc": NOW + timedelta(hours=3),
        "score_a": 0,
        "score_b": 0,
        "updated_at_utc": NOW,
        "tier": "a",
        "league_name": "BLAST Slam",
    } | kw


@pytest.fixture
def client(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'a.db').as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)
    with engine.begin() as conn:
        normalize_heroes_payload(
            conn, [{"id": h, "name": f"npc_dota_hero_h{h}", "localized_name": f"H{h}"} for h in range(1, 21)]
        )
        for i in range(30):
            normalize_match_detail(conn, _match(i, 1, 2, i % 3 != 0))
        conn.execute(
            insert(schedule_tournaments).values(
                tournament_key="ps:1",
                league_name="BLAST Slam",
                stage_name="Playoffs",
                tier="a",
                status="running",
                begin_at_utc=NOW,
                updated_at_utc=NOW,
            )
        )
        conn.execute(
            insert(fixtures),
            [
                _fx(
                    "ps:10",
                    team_a_id=1,
                    team_b_id=2,
                    team_a_name="One",
                    team_b_name="Two",
                    ps_team_a_id=101,
                    ps_team_b_id=102,
                ),
                _fx(
                    "ps:11",
                    team_a_id=1,
                    team_b_id=None,
                    team_a_name="One",
                    team_b_name="old blood",
                    ps_team_a_id=101,
                    ps_team_b_id=103,
                ),
                _fx(
                    "ps:12",
                    best_of=5,
                    team_a_id=None,
                    team_b_id=None,
                    team_a_name=None,
                    team_b_name=None,
                    ps_team_a_id=None,
                    ps_team_b_id=None,
                ),
            ],
        )
        conn.execute(
            insert(team_links).values(
                ps_team_id=103,
                ps_name="old blood",
                team_id=None,
                updated_at_utc=NOW,
                suggestions_json=json.dumps([{"team_id": 2, "name": "Team 2", "games": 30}]),
            )
        )
    predictor = LiveDraftPredictor(engine=engine, model_path=tmp_path / "x", predraft_model_path=tmp_path / "y")
    runner = JobRunner(engine, PredictionService(engine, predictor), live_fetch=list, schedule_enabled=False)
    app = create_app(engine=engine, runner=runner, password="pw", start_background=False)
    with TestClient(app) as c:
        yield c


def test_auth_required_except_healthz(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", auth=("admin", "nope")).status_code == 401
    assert client.get("/api/status", auth=AUTH).json()["phase"] == "ready"
    assert client.get("/", auth=AUTH).status_code == 200


def test_upcoming_fixtures_carry_predictions_and_flags(client):
    rows = {r["fixture_id"]: r for r in client.get("/api/fixtures?status=upcoming", auth=AUTH).json()}
    assert set(rows) == {"ps:10", "ps:11", "ps:12"}
    pred = rows["ps:10"]["prediction"]
    assert 0 < pred["p_series"] < 1 and pred["rows"][0]["label"] == "Match winner:"
    assert rows["ps:10"]["begin_at_utc"].endswith("Z")
    assert rows["ps:11"]["needs_mapping"] and "prediction" not in rows["ps:11"]
    assert rows["ps:12"]["tbd"] and rows["ps:12"]["team_a_name"] == "TBD"


def test_tournament_counts(client):
    (t,) = client.get("/api/tournaments?status=running", auth=AUTH).json()
    assert t["counts"] == {"total": 3, "predicted": 1, "tbd": 1, "needs_mapping": 1, "running": 0, "finished": 0}


def test_manual_link_enables_prediction(client):
    links = client.get("/api/links", auth=AUTH).json()
    assert links[0]["ps_team_id"] == 103 and links[0]["suggestions"][0]["team_id"] == 2
    assert client.put("/api/links/team/103", json={"team_id": 999}, auth=AUTH).status_code == 422
    assert client.put("/api/links/team/555", json={"team_id": 2}, auth=AUTH).status_code == 404
    assert client.put("/api/links/team/103", json={"team_id": 2}, auth=AUTH).json()["method"] == "manual"
    fx = client.get("/api/fixtures/ps:11", auth=AUTH).json()
    assert fx["team_b_id"] == 2 and "prediction" in fx and fx["history"] == []
    assert client.get("/api/links", auth=AUTH).json() == []


def test_record_and_team_search(client):
    assert client.get("/api/record", auth=AUTH).json()["overall"] == {"n": 0}
    found = client.get("/api/teams/search?q=team", auth=AUTH).json()
    assert {t["team_id"] for t in found} >= {1, 2}


def test_not_ready_returns_503(tmp_path):
    engine = create_engine("sqlite://")
    app = create_app(engine=engine, password="pw", start_background=False)
    with TestClient(app) as c:
        r = c.get("/api/fixtures", auth=AUTH)
        assert r.status_code == 503 and r.json()["detail"]["phase"] == "starting"


def test_password_is_mandatory(monkeypatch):
    monkeypatch.delenv("DOTA_UI_PASSWORD", raising=False)
    from dota_predictor.settings import get_settings

    get_settings.cache_clear()
    with pytest.raises(RuntimeError):
        create_app(engine=create_engine("sqlite://"), start_background=False)


def test_fixture_with_a_live_game_moves_from_upcoming_to_live(client):
    runner = client.app.state.dashboard.runner
    runner.live["ps:10"] = {"game": {"game_time": 120, "match_id": 1}, "prediction": None, "seen_at": NOW}
    live = [r["fixture_id"] for r in client.get("/api/fixtures?status=running", auth=AUTH).json()]
    upcoming = [r["fixture_id"] for r in client.get("/api/fixtures?status=upcoming", auth=AUTH).json()]
    assert live == ["ps:10"] and "ps:10" not in upcoming


def test_failed_startup_retries_until_it_succeeds(tmp_path, monkeypatch):
    import time

    import dota_predictor.service.app as app_mod

    engine = create_engine(f"sqlite:///{(tmp_path / 'b.db').as_posix()}", connect_args={"check_same_thread": False})
    calls = []

    def flaky_boot(state):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("OpenDota explorer HTTP 400: Error: Query read timeout")
        state.runner = JobRunner(engine, None, live_fetch=list, schedule_enabled=False)

    async def no_loop(self, name, start_delay=0):
        return None

    monkeypatch.setattr(app_mod, "_boot", flaky_boot)
    monkeypatch.setattr(app_mod, "BOOT_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(JobRunner, "loop", no_loop)
    app = create_app(engine=engine, password="pw", start_background=True)
    with TestClient(app) as c:
        for _ in range(100):
            if c.get("/healthz").json()["phase"] == "ready":
                break
            time.sleep(0.05)
        assert c.get("/healthz").json() == {"ok": True, "phase": "ready"}
    assert len(calls) == 2


def test_bookmaker_odds_are_saved_per_side_and_shown_on_fixtures(client):
    assert client.put("/api/fixtures/ps:404/odds", json={"odds_a": 1.8}, auth=AUTH).status_code == 404
    assert client.put("/api/fixtures/ps:10/odds", json={"odds_a": 0.9}, auth=AUTH).status_code == 422
    assert client.put("/api/fixtures/ps:10/odds", json={"odds_a": 1.8}, auth=AUTH).json()["odds_b"] is None
    saved = client.put("/api/fixtures/ps:10/odds", json={"odds_b": 2.1}, auth=AUTH).json()
    assert (saved["odds_a"], saved["odds_b"]) == (1.8, 2.1)  # a side left out keeps its price
    assert client.put("/api/fixtures/ps:10/odds", json={"odds_a": 0}, auth=AUTH).json()["odds_a"] is None  # 0 clears
    by_id = {f["fixture_id"]: f for f in client.get("/api/fixtures?status=upcoming", auth=AUTH).json()}
    assert by_id["ps:10"]["book_odds"] == {"a": None, "b": 2.1}
    assert by_id["ps:11"]["book_odds"] is None
