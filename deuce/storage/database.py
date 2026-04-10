"""SQLite engine and session factory."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from deuce.storage.schemas import Base


def default_sqlite_url() -> str:
    raw = os.getenv("DEUCE_DB_URL", "").strip()
    if raw:
        return raw
    root = Path(os.getenv("DEUCE_DATA_DIR", ".deuce")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / 'deuce.db'}"


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    url = url or default_sqlite_url()
    extra: dict = {}
    if ":memory:" in url:
        extra["connect_args"] = {"check_same_thread": False}
        extra["poolclass"] = StaticPool
    engine = create_engine(url, echo=echo, future=True, **extra)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk(dbapi_connection, connection_record):  # type: ignore[no-redef]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
