import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, insert, select

from dota_predictor.ingest.pandascore import fetch_all, finish_missing_tournaments, upsert_matches, upsert_tournaments
from dota_predictor.normalize.team_links import TeamIndex
from dota_predictor.storage.models import fixtures, games, schedule_tournaments, teams
from dota_predictor.storage.repository import init_db

FIX = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 10, tzinfo=UTC)


def load(name):
    return json.loads((FIX / f"pandascore_{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def conn():
    engine = create_engine("sqlite://")
    init_db(engine)
    with engine.begin() as c:
        named = [
            (9823272, "Team Yandex"),
            (7119388, "Team Spirit"),
            (9824702, "PARIVISION"),
            (9467224, "Aurora Gaming"),
            (2422576, "Cold BIood"),
        ]
        c.execute(
            insert(teams),
            [
                {"team_id": i, "canonical_name": n, "tag": "Aurora" if "Aurora" in n else None, "created_utc": NOW}
                for i, n in named
            ],
        )
        c.execute(
            insert(games),
            [
                {
                    "valve_match_id": k,
                    "radiant_team_id": i,
                    "dire_team_id": 1,
                    "start_utc": NOW,
                    "radiant_win": True,
                    "ingested_at_utc": NOW,
                }
                for k, (i, _) in enumerate(named)
            ],
        )
        yield c


def _rows(conn):
    return {r["fixture_id"]: r for r in conn.execute(select(fixtures)).mappings()}


def test_upcoming_matches_upsert_with_links_and_tbd(conn):
    upsert_matches(conn, load("upcoming"), TeamIndex.from_conn(conn))
    rows = _rows(conn)
    assert len(rows) == 12
    final = rows["ps:1709129"]
    assert (final["team_a_id"], final["team_b_id"]) == (9824702, 9467224)  # "Aurora" links via our tag
    assert (final["best_of"], final["status"], final["tier"]) == (3, "upcoming", "a")
    assert final["tournament_key"] == "ps:22039"
    tbd = rows["ps:1709131"]
    assert (tbd["team_a_name"], tbd["team_b_name"], tbd["best_of"]) == (None, None, 5)
    lookalike = rows["ps:1739626"]
    assert lookalike["team_b_name"] == "old blood" and lookalike["team_b_id"] is None


def test_rerun_is_idempotent(conn):
    for _ in range(2):
        upsert_matches(conn, load("upcoming"), TeamIndex.from_conn(conn))
    assert conn.execute(select(func.count()).select_from(fixtures)).scalar() == 12


def test_running_score_then_finished_winner(conn):
    m = copy.deepcopy(next(x for x in load("upcoming") if x["id"] == 1709129))
    m["status"] = "running"
    m["results"] = [{"team_id": 135712, "score": 1}, {"team_id": 133882, "score": 0}]
    upsert_matches(conn, [m], TeamIndex.from_conn(conn))
    row = _rows(conn)["ps:1709129"]
    assert (row["status"], row["score_a"], row["score_b"], row["winner"]) == ("running", 1, 0, None)
    upsert_matches(conn, load("past"), TeamIndex.from_conn(conn))
    lgd_aurora = _rows(conn)["ps:1709127"]
    assert (lgd_aurora["status"], lgd_aurora["score_a"], lgd_aurora["score_b"], lgd_aurora["winner"]) == (
        "finished",
        0,
        2,
        "b",
    )


def test_tournaments_upsert_and_finish(conn):
    upsert_tournaments(conn, load("tournaments_running"), "running")
    upsert_tournaments(conn, load("tournaments_running"), "running")
    assert conn.execute(select(func.count()).select_from(schedule_tournaments)).scalar() == 6
    blast = (
        conn.execute(select(schedule_tournaments).where(schedule_tournaments.c.tournament_key == "ps:22039"))
        .mappings()
        .one()
    )
    assert (blast["tier"], blast["league_name"], blast["serie_name"]) == ("a", "BLAST Slam", "Season 8 2026")
    finish_missing_tournaments(conn, {"ps:22039"})
    statuses = dict(conn.execute(select(schedule_tournaments.c.tournament_key, schedule_tournaments.c.status)).all())
    assert statuses["ps:22039"] == "running"
    assert sum(s == "finished" for s in statuses.values()) == 5


class _Resp:
    def __init__(self, rows, total):
        self._rows, self.headers, self.status_code = rows, {"x-total": str(total)}, 200

    def json(self):
        return self._rows

    def raise_for_status(self):
        return None


class _PagedClient:
    def __init__(self, total, per_page):
        self.total, self.per_page, self.calls = total, per_page, []

    def get(self, path, params=None):
        self.calls.append(params["page"])
        start = (params["page"] - 1) * self.per_page
        return _Resp(list(range(start, min(start + self.per_page, self.total))), self.total)


def test_fetch_all_follows_pages():
    client = _PagedClient(total=230, per_page=100)
    assert fetch_all(client, "/dota2/matches/upcoming") == list(range(230))
    assert client.calls == [1, 2, 3]
