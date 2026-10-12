import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, insert, select

from dota_predictor.normalize.team_links import TeamIndex, link_team, set_link
from dota_predictor.storage.models import fixtures, games, team_links, teams
from dota_predictor.storage.repository import init_db

NOW = datetime(2026, 10, 10, tzinfo=UTC)
TEAMS = [  # id, name, tag, games played
    (1, "Team Spirit", "TSpirit", 50),
    (2, "Aurora Gaming", "Aurora", 40),
    (3, "Cold BIood", "CB", 1),
    (4, "Royal Bear", "RB", 0),
    (5, "Team Falcons", "TF", 30),
    (6, "Tundra Falcons", "TF", 5),
    (7, "Nigma Galaxy", "NGX", 20),
]


@pytest.fixture
def conn():
    engine = create_engine("sqlite://")
    init_db(engine)
    with engine.begin() as c:
        c.execute(
            insert(teams), [{"team_id": i, "canonical_name": n, "tag": t, "created_utc": NOW} for i, n, t, _ in TEAMS]
        )
        rows, mid = [], 1
        for i, _, _, n in TEAMS:
            for _ in range(n):
                rows.append(
                    {
                        "valve_match_id": mid,
                        "radiant_team_id": i,
                        "dire_team_id": 99,
                        "start_utc": NOW,
                        "radiant_win": True,
                        "ingested_at_utc": NOW,
                    }
                )
                mid += 1
        c.execute(insert(games), rows)
        yield c


def _link(conn, ps_id, name, acronym=None):
    return link_team(conn, {"id": ps_id, "name": name, "acronym": acronym}, TeamIndex.from_conn(conn))


def test_exact_name_and_tag_link(conn):
    assert _link(conn, 10, "team spirit") == 1
    assert _link(conn, 11, "Aurora") == 2  # PandaScore name equals our tag
    assert _link(conn, 12, "Nigma", "NGX") == 7  # acronym equals our tag
    methods = dict(conn.execute(select(team_links.c.ps_team_id, team_links.c.method)).all())
    assert methods == {10: "exact_name", 11: "exact_tag", 12: "exact_tag"}


def test_core_name_strips_team_and_gaming(conn):
    assert _link(conn, 13, "Spirit") == 1


def test_lookalike_is_never_auto_linked(conn):
    assert _link(conn, 20, "old blood") is None
    row = conn.execute(select(team_links).where(team_links.c.ps_team_id == 20)).mappings().one()
    assert row["team_id"] is None
    assert [s["team_id"] for s in json.loads(row["suggestions_json"])] == [3]


def test_ambiguous_tag_and_zero_game_team_stay_unlinked(conn):
    assert _link(conn, 21, "Falcons Squad", "TF") is None
    assert _link(conn, 22, "Royal Bear") is None


def test_manual_link_survives_relinking_and_updates_fixtures(conn):
    _link(conn, 20, "old blood")
    conn.execute(
        insert(fixtures).values(
            fixture_id="ps:1",
            source="pandascore",
            best_of=3,
            status="upcoming",
            ps_team_a_id=20,
            ps_team_b_id=10,
            team_b_id=1,
            updated_at_utc=NOW,
        )
    )
    set_link(conn, 20, 3)
    assert _link(conn, 20, "old blood") == 3
    assert conn.execute(select(fixtures.c.team_a_id)).scalar() == 3


def test_shared_tag_links_the_only_recently_active_team(conn):
    from datetime import timedelta

    conn.execute(
        insert(teams),
        [
            {"team_id": 30, "canonical_name": "Nova Gaming", "tag": "Nova", "created_utc": NOW},
            {"team_id": 31, "canonical_name": "Nova.1xBet", "tag": "Nova", "created_utc": NOW},
        ],
    )
    old = datetime.now(UTC) - timedelta(days=400)
    conn.execute(
        insert(games),
        [
            {
                "valve_match_id": 9000 + k,
                "radiant_team_id": 31,
                "dire_team_id": 99,
                "start_utc": old,
                "radiant_win": True,
                "ingested_at_utc": NOW,
            }
            for k in range(50)
        ]
        + [
            {
                "valve_match_id": 9100,
                "radiant_team_id": 30,
                "dire_team_id": 99,
                "start_utc": datetime.now(UTC),
                "radiant_win": True,
                "ingested_at_utc": NOW,
            }
        ],
    )
    assert _link(conn, 40, "Nova", "NOV") == 30
    assert _link(conn, 41, "Falcons Squad", "TF") is None  # both "TF" teams active: still ambiguous


def test_quiet_team_with_little_history_waits_for_a_click(conn):
    from datetime import timedelta

    from sqlalchemy import update

    # "Cresent" once matched a 3-game id silent since August while the real team played on as "Crescent"
    old = datetime.now(UTC) - timedelta(days=45)
    conn.execute(update(games).values(start_utc=old).where(games.c.radiant_team_id.in_([1, 6])))
    assert _link(conn, 13, "Tundra Falcons") is None
    row = conn.execute(select(team_links).where(team_links.c.ps_team_id == 13)).mappings().one()
    assert json.loads(row["suggestions_json"])[0]["team_id"] == 6
    assert _link(conn, 14, "Team Spirit") == 1  # an established team keeps linking through a quiet spell
