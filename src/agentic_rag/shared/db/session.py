from collections.abc import AsyncIterator, Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from agentic_rag.shared.config import settings


_async_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None
_sync_engine: Engine | None = None
_sync_session_factory: sessionmaker[Session] | None = None


def _pool_options(database_url: str) -> dict[str, int]:
    if database_url.startswith("sqlite"):
        return {}
    return {
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
    }


def get_async_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.database_url,
            echo=settings.database_echo,
            pool_pre_ping=True,
            **_pool_options(settings.database_url),
        )
    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            get_async_engine(),
            autoflush=False,
            expire_on_commit=False,
        )
    return _async_session_factory


async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with get_async_session_factory()() as session:
        yield session


def get_sync_engine() -> Engine:
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(
            settings.sync_database_url,
            echo=settings.database_echo,
            pool_pre_ping=True,
            **_pool_options(settings.sync_database_url),
        )
    return _sync_engine


def get_sync_session_factory() -> sessionmaker[Session]:
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(),
            autoflush=False,
            expire_on_commit=False,
        )
    return _sync_session_factory


def get_session() -> Iterator[Session]:
    session = get_sync_session_factory()()
    try:
        yield session
    finally:
        session.close()


async def ping_database() -> None:
    async with get_async_engine().connect() as connection:
        await connection.execute(text("SELECT 1"))
