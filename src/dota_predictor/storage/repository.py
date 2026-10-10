from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Engine,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Connection

from dota_predictor.logging_config import get_logger
from dota_predictor.settings import get_settings
from dota_predictor.storage.models import SCHEMA_VERSION, metadata, providers

logger = get_logger("storage.repository")

_engine: Engine | None = None

schema_migrations = Table(
    "schema_migrations",
    metadata,
    Column("version", Text, primary_key=True),
    Column("applied_at_utc", DateTime(timezone=True), nullable=False, default=func.now()),
)


def get_engine(database_url: str | None = None) -> Engine:
    global _engine
    if _engine is not None and database_url is None:
        return _engine

    url = database_url or get_settings().database_url
    if url.startswith("sqlite"):
        # Ensure parent folder exists for SQLite file
        db_path_str = url.replace("sqlite:///", "")
        db_path = Path(db_path_str)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(url, echo=False, connect_args={"check_same_thread": False})
    else:
        eng = create_engine(url, echo=False, pool_pre_ping=True)

    if database_url is None:
        _engine = eng
    return eng


@contextmanager
def get_connection(engine: Engine | None = None) -> Generator[Connection, None, None]:
    eng = engine or get_engine()
    with eng.begin() as conn:
        yield conn


def init_db(engine: Engine | None = None) -> None:
    eng = engine or get_engine()
    logger.info("Initializing database schema", engine=str(eng.url))
    metadata.create_all(eng)

    with eng.begin() as conn:
        # Check and record migration
        res = conn.execute(select(schema_migrations).where(schema_migrations.c.version == SCHEMA_VERSION)).first()
        if not res:
            conn.execute(
                insert(schema_migrations).values(
                    version=SCHEMA_VERSION,
                    applied_at_utc=func.now(),
                )
            )
            logger.info("Recorded schema migration", version=SCHEMA_VERSION)

        # Seed default providers from config/providers.yaml if available
        settings = get_settings()
        prov_cfg = settings.providers_config()
        for prov_name, prov_data in prov_cfg.items():
            existing = conn.execute(select(providers).where(providers.c.name == prov_name)).first()
            if not existing:
                conn.execute(
                    insert(providers).values(
                        name=prov_name,
                        policy_version="1.0",
                        notes=prov_data.get("policy_notes", "").strip(),
                    )
                )
                logger.info("Seeded provider", provider=prov_name)
