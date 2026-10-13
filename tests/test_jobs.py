from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert, select

from dota_predictor.normalize.entities import normalize_heroes_payload
from dota_predictor.normalize.games import normalize_match_detail
from dota_predictor.predict.live_draft import LiveDraftPredictor
from dota_predictor.predict.service import PredictionService
from dota_predictor.service.coverage import coverage_alerts
from dota_predictor.service.jobs import JobRunner
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import (
    fixtures,
    games,
    prediction_results,
    predictions,
    pro_match_index,
    schedule_tournaments,
    team_links,
    tournaments,
)
from dota_predictor.storage.repository import init_db

NOW = datetime.now(UTC)


def _match(i, r, d, radiant_win):
    accounts = [r * 10 + k for k in range(5)] + [d * 10 + k for k in range(5)]
    return {
        "match_id": 80000 + i,
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


def _live(heroes=True, series_id=777):
    def side(team, team_no, offset):
        return [
            {"account_id": team * 10 + k, "hero_id": (offset + k) if heroes else 0, "team": team_no} for k in range(5)
        ]

    return [
        {
            "league_id": 15000,
            "series_id": series_id,
            "match_id": 9001,
            "team_id_radiant": 2,
            "team_id_dire": 1,
            "team_name_radiant": "Two",
            "team_name_dire": "One",
            "activate_time": int(NOW.timestamp()),
            "players": side(2, 0, 1) + side(1, 1, 6) + [{"account_id": 5, "hero_id": 0, "team": 2}],
        },
        {"league_id": 0, "team_id_radiant": 0, "team_id_dire": 0, "players": []},  # pub game: ignored
    ]


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'j.db').as_posix()}")
    init_db(engine)
    with engine.begin() as conn:
        normalize_heroes_payload(
            conn, [{"id": h, "name": f"npc_dota_hero_h{h}", "localized_name": f"Hero{h}"} for h in range(1, 21)]
        )
        for i in range(30):
            normalize_match_detail(conn, _match(i, 1, 2, i % 3 != 0))
    return engine


def _runner(engine, tmp_path, payload):
    predictor = LiveDraftPredictor(engine=engine, model_path=tmp_path / "x", predraft_model_path=tmp_path / "y")
    return JobRunner(engine, PredictionService(engine, predictor), live_fetch=lambda: payload, schedule_enabled=False)


def _kinds(engine):
    with engine.connect() as c:
        return sorted(
            (r.target_ref, r.prediction_type)
            for r in c.execute(select(predictions.c.target_ref, predictions.c.prediction_type))
        )


def test_opendota_only_fixture_gets_draft_and_locked_predictions(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    assert runner.status["running"].last_error is None
    with engine.connect() as c:
        fx = c.execute(select(fixtures)).mappings().one()
    assert (fx["fixture_id"], fx["source"], fx["team_a_id"], fx["status"]) == ("od:777", "opendota", 2, "running")
    assert _kinds(engine) == [("od:777", "live_draft"), ("od:777", "locked")]
    assert runner.live["od:777"]["prediction"]["map_number"] == 1
    runner.run("running")  # second poll writes nothing new
    assert len(_kinds(engine)) == 2


def test_partial_draft_locks_but_waits_for_heroes(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live(heroes=False))
    runner.run("running")
    assert _kinds(engine) == [("od:777", "locked")]
    assert runner.live["od:777"]["prediction"] is None


def test_live_game_attaches_to_scheduled_fixture(engine, tmp_path):
    with engine.begin() as conn:
        conn.execute(
            insert(fixtures).values(
                fixture_id="ps:5",
                source="pandascore",
                best_of=3,
                status="upcoming",
                team_a_id=1,
                team_b_id=2,
                begin_at_utc=NOW - timedelta(minutes=10),
                score_a=0,
                score_b=0,
                updated_at_utc=NOW,
            )
        )
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        ids = [r[0] for r in c.execute(select(fixtures.c.fixture_id))]
        od_series = c.execute(select(fixtures.c.od_series_id)).scalar()
    assert ids == ["ps:5"] and od_series == 777
    pred = runner.live["ps:5"]["prediction"]
    assert pred["a_side"] == "dire"  # team 1 is team A and plays dire in this game
    assert _kinds(engine) == [("ps:5", "live_draft"), ("ps:5", "locked")]


def test_failing_fetch_is_recorded_not_raised(engine, tmp_path):
    runner = _runner(engine, tmp_path, [])
    runner.live_fetch = lambda: (_ for _ in ()).throw(RuntimeError("opendota down"))
    runner.run("running")
    assert "opendota down" in runner.status["running"].last_error


def test_od_series_settles_and_grades(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.begin() as conn:
        conn.execute(
            insert(pro_match_index),
            [
                {  # the live game 9001 is already queued without a result and must not count
                    "match_id": 9002 + k,
                    "series_id": 777,
                    "series_type": 1,
                    "radiant_team_id": 2,
                    "dire_team_id": 1,
                    "radiant_win": True,
                    "start_time_utc": NOW,
                    "first_seen_utc": NOW,
                    "last_seen_utc": NOW,
                }
                for k in range(2)
            ],
        )
    runner.settle_and_check(NOW)
    with engine.connect() as c:
        fx = c.execute(select(fixtures)).mappings().one()
        results = [r[0] for r in c.execute(select(prediction_results.c.result_status))]
    assert (fx["status"], fx["score_a"], fx["score_b"], fx["winner"], fx["best_of"]) == ("finished", 2, 0, "a", 3)
    assert len(results) == 1 and results[0] in ("won", "lost")
    with engine.connect() as c:
        record = runner.service.record(c)
    assert record["ok"]["n"] == 1 and record["low_data"] == {"n": 0}  # both teams have 30 games on record


def test_every_live_game_is_queued_for_ingestion_and_does_not_settle_before_its_result(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        row = c.execute(select(pro_match_index).where(pro_match_index.c.match_id == 9001)).mappings().one()
    assert (row["detail_status"], row["leagueid"], row["series_id"], row["radiant_team_id"], row["dire_team_id"]) == (
        "pending",
        15000,
        777,
        2,
        1,
    )
    runner.settle_and_check(NOW)
    with engine.connect() as c:
        assert c.execute(select(fixtures.c.status)).scalar() == "running"  # a game without a result is not a win


class _OpenDota:
    """REST stand-in: knows one finished game, 404s on the rest, and records parse requests."""

    def __init__(self, known):
        self.known, self.requested, self.gets = known, [], []

    def get(self, path, params=None):
        self.gets.append(path)
        mid = int(path.rsplit("/", 1)[1])
        return _Resp(200, self.known[mid]) if mid in self.known else _Resp(404, {"error": "Not Found"})

    def post(self, path):
        self.requested.append(int(path.rsplit("/", 1)[1]))
        return _Resp(200, {"job": {"jobId": 1}})

    def close(self):
        pass


class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def _index_row(match_id, status, error=None, age=timedelta(hours=4)):
    return {
        "match_id": match_id,
        "detail_status": status,
        "detail_error": error,
        "leagueid": 20124,
        "radiant_team_id": 7,
        "dire_team_id": 8,
        "start_time_utc": NOW - age,
        "first_seen_utc": NOW,
        "last_seen_utc": NOW,
    }


def test_results_recovers_games_the_explorer_lacks(engine, tmp_path, monkeypatch):
    """A game in a league OpenDota does not track as pro: fetched by id, or requested from Valve on a miss."""
    import dota_predictor.ingest.opendota as od
    import dota_predictor.service.jobs as jobs

    game = _match(0, 7, 8, True) | {"match_id": 90001, "leagueid": 20124, "league": {"name": "UDL", "tier": "excluded"}}
    fake = _OpenDota({90001: game})
    monkeypatch.setattr(od, "RateLimitedClient", lambda *a, **k: fake)
    monkeypatch.setattr(get_settings(), "raw_store_dir", tmp_path / "raw")
    for name in ("sync_explorer_match_index", "sync_explorer_player_matches", "sync_explorer_teams"):
        monkeypatch.setattr(jobs, name, lambda *a, **k: {})
    monkeypatch.setattr(jobs, "refresh_reference", lambda *a, **k: {})
    with engine.begin() as conn:
        conn.execute(
            insert(pro_match_index),
            [
                _index_row(90001, "unavailable", "explorer: no player rows"),  # finished 4 h ago, OpenDota has it
                _index_row(90002, "unavailable", "explorer: no player rows"),  # OpenDota never saw it: ask for it
                _index_row(90003, "requested"),  # asked for last cycle, still not there: keep waiting
                _index_row(90004, "unavailable", "explorer: no player rows", age=timedelta(hours=1)),  # too fresh
            ],
        )
    runner = _runner(engine, tmp_path, [])
    runner.run("results")
    assert runner.status["results"].last_error is None, runner.status["results"].last_error
    with engine.connect() as c:
        status = dict(c.execute(select(pro_match_index.c.match_id, pro_match_index.c.detail_status)).all())
        result = c.execute(select(pro_match_index.c.radiant_win).where(pro_match_index.c.match_id == 90001)).scalar()
        stored = c.execute(
            select(games.c.radiant_team_id, games.c.dire_team_id).where(games.c.valve_match_id == 90001)
        ).one()
    assert (status[90001], status[90002], status[90003]) == ("fetched", "requested", "requested")
    assert status[90004] != "fetched" and "/matches/90004" not in fake.gets
    assert result is True and tuple(stored) == (7, 8)
    assert fake.requested == [90002]
    assert runner.service._games(7) == 1  # the new team is known to the ratings right away


def test_coverage_alerts(engine):
    with engine.begin() as conn:
        conn.execute(insert(tournaments).values(leagueid=15000, title="Big League", tier="premium"))
        conn.execute(
            insert(pro_match_index).values(
                match_id=1,
                leagueid=15000,
                league_name="Big League",
                radiant_team_id=1,
                dire_team_id=2,
                radiant_win=True,
                start_time_utc=NOW - timedelta(hours=5),
                first_seen_utc=NOW,
                last_seen_utc=NOW,
            )
        )
        conn.execute(
            insert(schedule_tournaments).values(
                tournament_key="ps:1",
                league_name="BLAST Slam",
                stage_name="Playoffs",
                tier="a",
                status="running",
                updated_at_utc=NOW,
            )
        )
        kinds = sorted(a["kind"] for a in coverage_alerts(conn, NOW, schedule_enabled=True))
        assert kinds == ["missing_from_schedule", "no_matches_listed"]
        conn.execute(
            insert(fixtures).values(
                fixture_id="ps:9",
                source="pandascore",
                tournament_key="ps:1",
                best_of=3,
                status="upcoming",
                team_a_id=2,
                team_b_id=1,
                begin_at_utc=NOW + timedelta(hours=2),
                score_a=0,
                score_b=0,
                updated_at_utc=NOW,
            )
        )
        assert coverage_alerts(conn, NOW, schedule_enabled=True) == []
        assert coverage_alerts(conn, NOW, schedule_enabled=False)[0]["kind"] == "schedule_disabled"


def test_ended_games_in_the_live_feed_are_ignored(engine, tmp_path):
    payload = _live()
    payload[0]["deactivate_time"] = int(NOW.timestamp())
    runner = _runner(engine, tmp_path, payload)
    runner.run("running")
    with engine.connect() as c:
        assert c.execute(select(fixtures)).first() is None


def _ps_fixture(conn, **kw):
    base = {
        "fixture_id": "ps:7",
        "source": "pandascore",
        "best_of": 3,
        "status": "upcoming",
        "team_a_id": None,
        "team_b_id": None,
        "team_a_name": "One",
        "team_b_name": "Two",
        "ps_team_a_id": 501,
        "ps_team_b_id": 502,
        "begin_at_utc": NOW - timedelta(minutes=5),
        "score_a": 0,
        "score_b": 0,
        "updated_at_utc": NOW,
    }
    conn.execute(insert(fixtures).values(base | kw))


def test_live_game_links_unlinked_schedule_teams_by_exact_name(engine, tmp_path):
    with engine.begin() as conn:
        _ps_fixture(conn)
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        fx = c.execute(select(fixtures)).mappings().all()
        links = dict(c.execute(select(team_links.c.ps_team_id, team_links.c.team_id)).all())
        methods = {r[0] for r in c.execute(select(team_links.c.method))}
    assert [f["fixture_id"] for f in fx] == ["ps:7"]
    assert (fx[0]["team_a_id"], fx[0]["team_b_id"]) == (1, 2)
    assert links == {501: 1, 502: 2} and methods == {"live_match"}
    assert "ps:7" in runner.live


def test_live_game_for_a_finished_match_is_ignored(engine, tmp_path):
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=2, status="finished", winner="a")
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        assert [r[0] for r in c.execute(select(fixtures.c.fixture_id))] == ["ps:7"]
    assert runner.live == {}


def test_opendota_duplicate_is_canceled_once_the_schedule_has_the_match(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=2)
    runner.run("running")
    with engine.connect() as c:
        status = dict(c.execute(select(fixtures.c.fixture_id, fixtures.c.status)).all())
    assert status == {"od:777": "canceled", "ps:7": "upcoming"}
    assert set(runner.live) == {"ps:7"}


def test_live_game_corrects_an_automatic_link_to_a_duplicate_team_id(engine, tmp_path):
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=55, team_b_name="Too")  # 55: a look-alike id of team 2
        conn.execute(
            insert(team_links).values(
                ps_team_id=502, ps_name="Too", team_id=55, method="exact_name", updated_at_utc=NOW
            )
        )
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        fx = c.execute(select(fixtures)).mappings().all()
        link = c.execute(select(team_links).where(team_links.c.ps_team_id == 502)).mappings().one()
    assert [f["fixture_id"] for f in fx] == ["ps:7"] and fx[0]["team_b_id"] == 2
    assert (link["team_id"], link["method"]) == (2, "live_match")


def test_live_game_corrects_a_manual_link_when_the_scheduled_opponent_is_on_the_field(engine, tmp_path):
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=55, team_b_name="Too")  # 55: a hand-picked id that never plays
        conn.execute(
            insert(team_links).values(ps_team_id=502, ps_name="Too", team_id=55, method="manual", updated_at_utc=NOW)
        )
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        ids = sorted(r[0] for r in c.execute(select(fixtures.c.fixture_id)))
        team_b = c.execute(select(fixtures.c.team_b_id).where(fixtures.c.fixture_id == "ps:7")).scalar()
        link = c.execute(select(team_links).where(team_links.c.ps_team_id == 502)).mappings().one()
    assert ids == ["ps:7"] and team_b == 2
    assert (link["team_id"], link["method"]) == (2, "live_match")


def test_schedule_match_wins_over_an_earlier_opendota_copy_with_lookalike_ids(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")  # no schedule yet: OpenDota-only copy
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=55, team_b_name="Too")
        conn.execute(
            insert(team_links).values(
                ps_team_id=502, ps_name="Too", team_id=55, method="exact_name", updated_at_utc=NOW
            )
        )
    runner.run("running")
    with engine.connect() as c:
        status = dict(c.execute(select(fixtures.c.fixture_id, fixtures.c.status)).all())
        team_b = c.execute(select(fixtures.c.team_b_id).where(fixtures.c.fixture_id == "ps:7")).scalar()
    assert status == {"od:777": "canceled", "ps:7": "upcoming"} and team_b == 2
    assert set(runner.live) == {"ps:7"}


def test_lock_is_written_once_even_after_a_model_change(engine, tmp_path):
    runner = _runner(engine, tmp_path, _live(heroes=False))
    runner.run("running")
    runner.service.model_version = "retrained"  # a deploy with new model files must not re-lock a running match
    runner.run("running")
    assert _kinds(engine) == [("od:777", "locked")]


def test_no_lock_once_a_map_has_been_played(engine, tmp_path):
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=2, status="running", score_a=1)
    runner = _runner(engine, tmp_path, [])
    runner.run("running")
    assert ("ps:7", "locked") not in _kinds(engine)


def test_stale_schedule_fixture_is_canceled(engine, tmp_path):
    with engine.begin() as conn:
        _ps_fixture(conn, team_a_id=1, team_b_id=2, status="running", updated_at_utc=NOW - timedelta(hours=13))
    runner = _runner(engine, tmp_path, [])
    runner.run("running")
    with engine.connect() as c:
        assert c.execute(select(fixtures.c.status)).scalar() == "canceled"


def _same_players_under_new_ids(conn):
    """One game where ids 55 and 66 fielded exactly the players of teams 2 and 1: dead duplicate ids."""
    m = _match(40, 55, 66, True)
    for p, acc in zip(m["players"], [20, 21, 22, 23, 24, 10, 11, 12, 13, 14], strict=True):
        p["account_id"] = acc
    normalize_match_detail(conn, m)


def test_live_game_relinks_dead_duplicate_ids_by_their_players(engine, tmp_path):
    with engine.begin() as conn:
        _same_players_under_new_ids(conn)
        _ps_fixture(conn, team_a_id=55, team_b_id=66, team_a_name="Too", team_b_name="Won")  # no id or name matches
        conn.execute(
            insert(team_links),
            [
                {"ps_team_id": 501, "ps_name": "Too", "team_id": 55, "method": "exact_name", "updated_at_utc": NOW},
                {"ps_team_id": 502, "ps_name": "Won", "team_id": 66, "method": "exact_name", "updated_at_utc": NOW},
            ],
        )
    runner = _runner(engine, tmp_path, _live())
    runner.run("running")
    with engine.connect() as c:
        fx = c.execute(select(fixtures)).mappings().all()
        links = dict(c.execute(select(team_links.c.ps_team_id, team_links.c.team_id)).all())
    assert [f["fixture_id"] for f in fx] == ["ps:7"] and (fx[0]["team_a_id"], fx[0]["team_b_id"]) == (2, 1)
    assert links == {501: 2, 502: 1}


def test_a_refused_parse_request_is_retried_next_cycle(engine, tmp_path, monkeypatch):
    import dota_predictor.ingest.opendota as od
    import dota_predictor.service.jobs as jobs

    class Refusing(_OpenDota):
        def post(self, path):
            super().post(path)
            return _Resp(429, {"error": "rate limit"})

    fake = Refusing({})
    monkeypatch.setattr(od, "RateLimitedClient", lambda *a, **k: fake)
    for name in ("sync_explorer_match_index", "sync_explorer_player_matches", "sync_explorer_teams"):
        monkeypatch.setattr(jobs, name, lambda *a, **k: {})
    monkeypatch.setattr(jobs, "refresh_reference", lambda *a, **k: {})
    with engine.begin() as conn:
        conn.execute(insert(pro_match_index), [_index_row(90002, "unavailable", "explorer: no player rows")])
    runner = _runner(engine, tmp_path, [])
    runner.run("results")
    with engine.connect() as c:
        status = c.execute(select(pro_match_index.c.detail_status).where(pro_match_index.c.match_id == 90002)).scalar()
    assert fake.requested == [90002] and status != "requested"


def _stub_explorer(monkeypatch, fake, tmp_path):
    import dota_predictor.ingest.opendota as od
    import dota_predictor.service.jobs as jobs

    monkeypatch.setattr(od, "RateLimitedClient", lambda *a, **k: fake)
    monkeypatch.setattr(get_settings(), "raw_store_dir", tmp_path / "raw")
    for name in ("sync_explorer_match_index", "sync_explorer_player_matches", "sync_explorer_teams"):
        monkeypatch.setattr(jobs, name, lambda *a, **k: {})
    monkeypatch.setattr(jobs, "refresh_reference", lambda *a, **k: {})


def test_a_game_is_fetched_as_soon_as_it_leaves_the_feed(engine, tmp_path, monkeypatch):
    finished = _match(0, 2, 1, True) | {
        "match_id": 9001,
        "leagueid": 15000,
        "league": {"name": "L", "tier": "excluded"},
    }
    fake = _OpenDota({9001: finished})
    _stub_explorer(monkeypatch, fake, tmp_path)
    feed = [_live(), [], []]
    runner = _runner(engine, tmp_path, None)
    runner.live_fetch = lambda: feed.pop(0)
    runner.run("running")
    runner.run("running")
    assert fake.gets == []  # gone for one poll may just be the feed blinking
    runner.run("running")
    assert runner.status["running"].last_error is None, runner.status["running"].last_error
    assert fake.gets == ["/matches/9001"]
    assert runner.service._games(2) == 31  # folded into the ratings in the same cycle


def test_a_requested_game_is_asked_again_after_an_hour_but_at_most_three_times(engine, tmp_path, monkeypatch):
    fake = _OpenDota({})
    _stub_explorer(monkeypatch, fake, tmp_path)
    with engine.begin() as conn:
        conn.execute(
            insert(pro_match_index),
            [
                _index_row(90001, "requested", "requests=1", age=timedelta(minutes=30))
                | {"last_seen_utc": NOW - timedelta(hours=2)},
                _index_row(90002, "requested", "requests=1") | {"last_seen_utc": NOW - timedelta(minutes=20)},
                _index_row(90003, "requested", "requests=3") | {"last_seen_utc": NOW - timedelta(hours=5)},
            ],
        )
    runner = _runner(engine, tmp_path, [])
    runner.run("results")
    assert runner.status["results"].last_error is None, runner.status["results"].last_error
    with engine.connect() as c:
        errors = dict(c.execute(select(pro_match_index.c.match_id, pro_match_index.c.detail_error)).all())
    assert fake.requested == [90001]
    assert errors == {90001: "requests=2", 90002: "requests=1", 90003: "requests=3"}


def test_running_cycle_grades_a_finished_series(engine, tmp_path, monkeypatch):
    _stub_explorer(monkeypatch, _OpenDota({}), tmp_path)
    feed = [_live(), []]
    runner = _runner(engine, tmp_path, None)
    runner.live_fetch = lambda: feed.pop(0)
    runner.run("running")
    with engine.begin() as conn:
        conn.execute(
            insert(pro_match_index),
            [
                {
                    "match_id": 9002 + k,
                    "series_id": 777,
                    "series_type": 1,
                    "radiant_team_id": 2,
                    "dire_team_id": 1,
                    "radiant_win": True,
                    "start_time_utc": NOW,
                    "first_seen_utc": NOW,
                    "last_seen_utc": NOW,
                }
                for k in range(2)
            ],
        )
    runner.run("running")
    with engine.connect() as c:
        assert c.execute(select(fixtures.c.status)).scalar() == "finished"
        assert c.execute(select(prediction_results.c.result_status)).scalar() in ("won", "lost")
