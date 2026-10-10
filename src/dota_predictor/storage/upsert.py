"""INSERT ... ON CONFLICT that compiles on the connection's own dialect.

SQLite (tests, local runs) and Postgres (server) share the same upsert API,
but each dialect's construct only compiles on its own backend.
"""

from __future__ import annotations

from sqlalchemy import Table
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Connection


def dialect_insert(conn: Connection, table: Table):  # noqa: ANN201 - dialect-specific Insert subclass
    return (postgresql if conn.dialect.name == "postgresql" else sqlite).insert(table)
