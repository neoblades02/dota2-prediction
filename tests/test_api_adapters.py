from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import create_engine, select

from dota_predictor.ingest.opendota import sync_match_details, sync_pro_match_index
from dota_predictor.storage.models import pro_match_index
from dota_predictor.storage.repository import init_db


@pytest.fixture
def test_engine(tmp_path):
    db_file = tmp_path / "test_adapters.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)
    return engine


def test_sync_pro_match_index_mocked(test_engine, monkeypatch):
    mock_matches = [
        {
            "match_id": 9000000001,
            "duration": 2100,
            "start_time": 1720000000,
            "radiant_team_id": 11,
            "radiant_name": "Team Liquid",
            "dire_team_id": 22,
            "dire_name": "Team Spirit",
            "leagueid": 100,
            "league_name": "The International",
            "series_id": 55,
            "series_type": 1,
            "radiant_score": 35,
            "dire_score": 20,
            "radiant_win": True,
            "version": "60",
        }
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=mock_matches)

    monkeypatch.setattr(
        "dota_predictor.ingest.opendota.RateLimitedClient",
        lambda *args, **kwargs: type(
            "MockClient",
            (),
            {
                "get": lambda self, ep, params=None: httpx.Response(200, json=mock_matches),
                "close": lambda self: None,
            },
        )(),
    )

    res = sync_pro_match_index(test_engine, max_pages=1, resume=False)
    assert res["pages_fetched"] == 1
    assert res["total_matches_upserted"] == 1

    with test_engine.begin() as conn:
        row = conn.execute(select(pro_match_index).where(pro_match_index.c.match_id == 9000000001)).first()
        assert row is not None
        assert row.radiant_name == "Team Liquid"
        assert row.radiant_win is True
        assert row.detail_status == "pending"


def test_sync_match_details_mocked(test_engine, monkeypatch):
    # Seed a pending match
    with test_engine.begin() as conn:
        conn.execute(
            pro_match_index.insert().values(
                match_id=9000000002,
                detail_status="pending",
                first_seen_utc=datetime.now(UTC),
                last_seen_utc=datetime.now(UTC),
            )
        )

    mock_detail = {
        "match_id": 9000000002,
        "radiant_win": False,
        "players": [{"account_id": 1234, "hero_id": 1}],
    }

    monkeypatch.setattr(
        "dota_predictor.ingest.opendota.RateLimitedClient",
        lambda *args, **kwargs: type(
            "MockClient",
            (),
            {
                "get": lambda self, ep, params=None: httpx.Response(200, json=mock_detail),
                "close": lambda self: None,
            },
        )(),
    )

    res = sync_match_details(test_engine, limit=1)
    assert res["fetched"] == 1

    with test_engine.begin() as conn:
        row = conn.execute(select(pro_match_index).where(pro_match_index.c.match_id == 9000000002)).first()
        assert row.detail_status == "fetched"
