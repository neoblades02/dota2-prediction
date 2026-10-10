from sqlalchemy import create_engine, func, select

from dota_predictor.pipeline import normalize_reference
from dota_predictor.storage.models import metadata, patches
from dota_predictor.storage.raw_store import save_raw_payload

DC = [
    {"id": 58, "name": "7.39", "date": "2025-05-22T00:00:00Z"},
    {"id": 59, "name": "7.40", "date": "2025-08-01T00:00:00Z"},
]
VALVE = {
    "patches": [
        {"patch_number": "7.39", "patch_name": "7.39", "patch_timestamp": 1747900000},
        {"patch_number": "7.39b", "patch_name": "7.39b", "patch_timestamp": 1748500000},
    ]
}


def test_fresh_db_normalizes_twice_without_duplicate_versions(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    metadata.create_all(engine)
    with engine.begin() as conn:
        save_raw_payload(conn, "opendota", "/dotaconstants/patch", "k", DC)
        save_raw_payload(conn, "valve_patches", "/datafeed/patchnoteslist", "k", VALVE)

    normalize_reference(engine)
    normalize_reference(engine)  # second boot / daily refresh must not raise MultipleResultsFound

    with engine.connect() as conn:
        dupes = conn.execute(select(patches.c.version).group_by(patches.c.version).having(func.count() > 1)).all()
        assert dupes == []
        # dotaconstants id is canonical; Valve only adds letter patches it alone knows
        assert conn.execute(select(patches.c.patch_id).where(patches.c.version == "7.39")).scalar_one() == 58
        assert conn.execute(select(patches.c.patch_id).where(patches.c.version == "7.39b")).scalar_one() >= 1000
