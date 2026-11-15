"""SQLAlchemy engine and session factory.

``DEUCE_DB_URL`` selects the database. Postgres is the product path. When the
variable is unset, local runs and tests use SQLite.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from deuce.storage.schemas import Base


def default_sqlite_url() -> str:
    from deuce.config.settings import database_url

    return database_url()


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    url = url or default_sqlite_url()
    extra: dict = {}
    if url.startswith("sqlite") and ":memory:" in url:
        extra["connect_args"] = {"check_same_thread": False}
        extra["poolclass"] = StaticPool
    elif not url.startswith("sqlite"):
        extra["pool_pre_ping"] = True
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
