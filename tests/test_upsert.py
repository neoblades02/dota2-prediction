from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql, sqlite

from dota_predictor.settings import normalize_database_url
from dota_predictor.storage.models import pro_match_index, teams
from dota_predictor.storage.upsert import dialect_insert

SRC = Path(__file__).resolve().parents[1] / "src" / "dota_predictor"


class _Conn:
    def __init__(self, dialect):
        self.dialect = dialect


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.psycopg.dialect()])
def test_upserts_compile_on_both_dialects(dialect):
    conn = _Conn(dialect)
    nothing = dialect_insert(conn, teams).values(team_id=1, canonical_name="x").on_conflict_do_nothing()
    update = (
        dialect_insert(conn, pro_match_index)
        .values(match_id=1, leagueid=2)
        .on_conflict_do_update(index_elements=["match_id"], set_={"leagueid": 2})
    )
    for stmt in (nothing, update):
        assert "ON CONFLICT" in str(stmt.compile(dialect=dialect))


def test_no_module_uses_sqlite_only_insert():
    offenders = [p.name for p in SRC.rglob("*.py") if "dialects.sqlite import insert" in p.read_text(encoding="utf-8")]
    assert offenders == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
        ("postgres://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u@h/db", "postgresql+psycopg://u@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_database_url_driver_rewrite(raw, expected):
    assert normalize_database_url(raw) == expected


def test_settings_accept_plain_database_url(monkeypatch):
    from dota_predictor.settings import Settings

    monkeypatch.delenv("DOTA_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:1/db")
    assert Settings(_env_file=None).database_url == "postgresql+psycopg://u:p@h:1/db"
    monkeypatch.setenv("DOTA_DATABASE_URL", "sqlite:///override.db")
    assert Settings(_env_file=None).database_url == "sqlite:///override.db"
