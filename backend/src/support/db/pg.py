"""Postgres connection pool for the API's tables (the agent's memory has its own pool, api/runner.py)."""

import threading
from collections.abc import Mapping, Sequence
from typing import Any, LiteralString, cast

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from support.settings import settings

Params = Sequence[Any] | Mapping[str, Any] | None

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def pool() -> ConnectionPool:
    global _pool
    with _lock:
        if _pool is None:
            if not settings.database_url:
                raise RuntimeError("DATABASE_URL is not set (see .env.example)")
            _pool = ConnectionPool(
                settings.database_url, min_size=1, max_size=8, kwargs={"row_factory": dict_row}, open=False
            )
            _pool.open(wait=True, timeout=30)
        return _pool


def close() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def connection():
    """`with pg.connection() as conn:` commits on success and rolls back on an exception."""
    return pool().connection()


def scalar(conn: Connection[Any], sql: str, params: Params = None) -> Any:
    """The first column of the first row, or None. SQL is written in this code; every value goes through `params`."""
    row = conn.execute(cast(LiteralString, sql), params).fetchone()
    return next(iter(row.values())) if row else None
