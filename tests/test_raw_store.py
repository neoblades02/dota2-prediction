import json

import pytest
from sqlalchemy import create_engine

from dota_predictor.ingest.base import get_sync_cursor, set_sync_cursor
from dota_predictor.settings import get_settings
from dota_predictor.storage.raw_store import compute_sha256, load_raw_payload, save_raw_payload
from dota_predictor.storage.repository import get_connection, init_db


@pytest.fixture
def test_engine(tmp_path):
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file.as_posix()}", connect_args={"check_same_thread": False})
    init_db(engine)
    return engine


def test_save_and_load_raw_payload(test_engine, tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "raw_store_dir", tmp_path / "raw")

    data = {"match_id": 123456, "radiant_win": True, "players": [{"hero_id": 1}]}
    with get_connection(test_engine) as conn:
        obj_id = save_raw_payload(
            conn=conn,
            provider="opendota",
            endpoint="/matches/123456",
            request_key="123456",
            payload=data,
        )
        assert obj_id is not None

        # Repeat save returns same ID (idempotent)
        obj_id2 = save_raw_payload(
            conn=conn,
            provider="opendota",
            endpoint="/matches/123456",
            request_key="123456",
            payload=data,
        )
        assert obj_id == obj_id2

    # Verify loading payload
    hash_str = compute_sha256(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))
    payload_file = tmp_path / "raw" / "opendota" / "matches_123456" / f"{hash_str}.json.gz"
    assert payload_file.exists()

    loaded = load_raw_payload(payload_file)
    assert loaded["match_id"] == 123456
    assert loaded["radiant_win"] is True


def test_sync_cursor_lifecycle(test_engine):
    with get_connection(test_engine) as conn:
        set_sync_cursor(conn, "opendota", "proMatches", cursor="1000", state="running")
        cur = get_sync_cursor(conn, "opendota", "proMatches")
        assert cur is not None
        assert cur["cursor"] == "1000"
        assert cur["state"] == "running"

        set_sync_cursor(conn, "opendota", "proMatches", cursor="900", state="done")
        cur2 = get_sync_cursor(conn, "opendota", "proMatches")
        assert cur2["cursor"] == "900"
        assert cur2["state"] == "done"
