# smartnotes/db.py
from __future__ import annotations
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy import inspect

from smartnotes.models import Base


async def ensure_schema(engine: AsyncEngine) -> None:
    """
    Create tables if the database is empty (no Alembic needed).
    Idempotent: does nothing if tables already exist.
    """
    async with engine.begin() as conn:
        def _get_tables(sync_conn):
            insp = inspect(sync_conn)
            return insp.get_table_names()

        existing = await conn.run_sync(_get_tables)
        if not existing:
            print("[db] creating tables...")
            await conn.run_sync(Base.metadata.create_all)


def make_engine(db_path: Path) -> tuple[AsyncEngine, async_sessionmaker]:
    """
    Build the async engine + session factory for the SQLite DB at db_path.
    """
    url = f"sqlite+aiosqlite:///{db_path}"
    eng = create_async_engine(url, future=True)
    session_factory = async_sessionmaker(eng, expire_on_commit=False)
    return eng, session_factory
