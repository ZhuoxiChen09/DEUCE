"""Interval scheduler for the historical import job.

The scheduler is off unless ``DEUCE_IMPORT_ENABLED=1``. It does not scrape
live scores.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from deuce.config.settings import import_interval_minutes, import_on_start
from deuce.ingestion.adapters import DataSourceAdapter, get_adapter
from deuce.ingestion.job import run_import

logger = logging.getLogger("deuce.ingestion")


def scheduled_import(
    session_factory: sessionmaker[Session],
    adapter: DataSourceAdapter | None = None,
) -> None:
    try:
        selected = adapter or get_adapter()
        outcome = run_import(session_factory=session_factory, adapter=selected)
        logger.info(
            "scheduled import finished status=%s loaded=%s",
            outcome.status,
            outcome.rows_loaded,
        )
    except Exception:
        logger.exception("scheduled import failed")


def build_import_scheduler(
    job: Callable[[], None],
    *,
    minutes: int | None = None,
    run_immediately: bool | None = None,
) -> Any:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    interval = minutes if minutes is not None else import_interval_minutes()
    immediate = import_on_start() if run_immediately is None else run_immediately
    scheduler = BackgroundScheduler(daemon=True, timezone="UTC")
    scheduler.add_job(
        job,
        IntervalTrigger(minutes=max(1, int(interval))),
        id="deuce-import",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc) if immediate else None,
        misfire_grace_time=600,
    )
    return scheduler
