"""PostgreSQL connection management for the FastAPI application."""

import os
from collections.abc import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    if not database_url.startswith("postgresql+asyncpg://"):
        raise RuntimeError("DATABASE_URL must use the postgresql+asyncpg driver")
    return database_url


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            get_database_url(),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async session factory used by repositories."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_database_session() -> AsyncIterator[AsyncSession]:
    """Provide one request-scoped database session."""
    async with get_session_factory()() as session:
        yield session


async def check_database_connection() -> dict[str, str]:
    async with get_engine().connect() as connection:
        postgres_version = (await connection.execute(text("SELECT version()"))).scalar_one()
        pgvector_version = (
            await connection.execute(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'"))
        ).scalar_one_or_none()
    if pgvector_version is None:
        raise RuntimeError("PostgreSQL is reachable, but the vector extension is not enabled")
    return {
        "status": "healthy",
        "postgres": postgres_version,
        "pgvector": pgvector_version,
    }


async def close_database_connection() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
