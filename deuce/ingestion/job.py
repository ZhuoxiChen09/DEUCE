"""Run one historical import and persist the data-quality decision."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from deuce.config.settings import database_url, dq_date_bounds, dq_mode, sqlite_auto_create
from deuce.ingestion.adapters import DataSourceAdapter, SourceLoadError, get_adapter
from deuce.ingestion.quality import (
    DQIssue,
    DQReport,
    NormalizedMatch,
    evaluate_records,
    issue_to_dict,
    schema_issues,
)
from deuce.storage.database import init_db, make_engine, make_session_factory, session_scope
from deuce.storage.schemas import HistoricalMatchRow, ImportRunRow

logger = logging.getLogger("deuce.ingestion")


@dataclass
class ImportOutcome:
    status: str
    source_id: str
    mode: str
    rows_seen: int
    rows_loaded: int
    rows_rejected: int
    message: str
    issues: list[DQIssue]
    run_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_id": self.source_id,
            "mode": self.mode,
            "rows_seen": self.rows_seen,
            "rows_loaded": self.rows_loaded,
            "rows_rejected": self.rows_rejected,
            "message": self.message,
            "issues": [issue_to_dict(issue) for issue in self.issues],
            "run_id": self.run_id,
        }


def run_import(
    *,
    session_factory: sessionmaker[Session] | None = None,
    adapter: DataSourceAdapter | None = None,
    mode: str | None = None,
) -> ImportOutcome:
    """Load one adapter, check quality, and block or flag a bad load.

    ``block`` (the default) writes no match rows when any error-level check
    fails. ``flag`` writes the valid rows and skips the rest. Schema failures
    and unreadable sources never write match rows. Every attempt is logged
    and stored in ``import_runs``.
    """
    selected_mode = (mode or dq_mode()).strip().lower()
    if selected_mode not in {"block", "flag"}:
        raise ValueError("mode must be 'block' or 'flag'")
    source = adapter or get_adapter()
    factory = session_factory or _default_session_factory()
    try:
        loaded = source.load()
    except SourceLoadError as exc:
        outcome = _failure(
            factory,
            source_id=source.source_id,
            mode=selected_mode,
            status="error",
            message=str(exc),
            issues=[],
        )
        _log_outcome(outcome)
        return outcome

    column_issues = schema_issues(loaded.fieldnames)
    if column_issues:
        outcome = _failure(
            factory,
            source_id=loaded.source_id,
            mode=selected_mode,
            status="blocked",
            message=column_issues[0].message,
            issues=column_issues,
            rows_seen=len(loaded.records),
        )
        _log_outcome(outcome)
        return outcome

    date_min, date_max = dq_date_bounds()
    report = evaluate_records(loaded.records, date_min=date_min, date_max=date_max)
    return _apply(factory, source_id=loaded.source_id, mode=selected_mode, report=report)


def _default_session_factory() -> sessionmaker[Session]:
    url = database_url()
    engine = make_engine(url)
    if sqlite_auto_create(url):
        init_db(engine)
    return make_session_factory(engine)


def _apply(
    factory: sessionmaker[Session],
    *,
    source_id: str,
    mode: str,
    report: DQReport,
) -> ImportOutcome:
    has_errors = bool(report.errors)
    has_warnings = any(issue.severity == "warning" for issue in report.issues)
    if has_errors and mode == "block":
        status = "blocked"
        to_load: list[NormalizedMatch] = []
    elif has_errors or has_warnings:
        status = "flagged"
        to_load = list(report.valid)
    else:
        status = "loaded"
        to_load = list(report.valid)

    rows_loaded = len(to_load)
    rows_rejected = report.rows_seen - rows_loaded
    if status == "blocked":
        message = f"blocked: {len(report.errors)} data-quality error(s); no rows written"
    elif status == "flagged":
        message = f"flagged: loaded {rows_loaded}, rejected {rows_rejected}"
    elif rows_loaded == 0:
        message = "loaded 0 row(s)"
    else:
        message = f"loaded {rows_loaded} row(s)"

    outcome = _persist(
        factory,
        source_id=source_id,
        mode=mode,
        status=status,
        message=message,
        issues=report.issues,
        rows_seen=report.rows_seen,
        rows_loaded=rows_loaded,
        rows_rejected=rows_rejected,
        matches=to_load,
    )
    _log_outcome(outcome)
    return outcome


def _failure(
    factory: sessionmaker[Session],
    *,
    source_id: str,
    mode: str,
    status: str,
    message: str,
    issues: list[DQIssue],
    rows_seen: int = 0,
) -> ImportOutcome:
    return _persist(
        factory,
        source_id=source_id,
        mode=mode,
        status=status,
        message=message,
        issues=issues,
        rows_seen=rows_seen,
        rows_loaded=0,
        rows_rejected=rows_seen,
        matches=[],
    )


def _persist(
    factory: sessionmaker[Session],
    *,
    source_id: str,
    mode: str,
    status: str,
    message: str,
    issues: list[DQIssue],
    rows_seen: int,
    rows_loaded: int,
    rows_rejected: int,
    matches: list[NormalizedMatch],
) -> ImportOutcome:
    started = datetime.utcnow()
    with session_scope(factory) as session:
        run = ImportRunRow(
            source_id=source_id,
            status=status,
            mode=mode,
            rows_seen=rows_seen,
            rows_loaded=rows_loaded,
            rows_rejected=rows_rejected,
            issues_json=json.dumps([issue_to_dict(issue) for issue in issues], ensure_ascii=False),
            message=message,
            started_at=started,
            finished_at=datetime.utcnow(),
        )
        session.add(run)
        session.flush()
        for match in matches:
            _upsert_match(session, match, source_id=source_id, import_run_id=run.id)
        run_id = run.id
    return ImportOutcome(
        status=status,
        source_id=source_id,
        mode=mode,
        rows_seen=rows_seen,
        rows_loaded=rows_loaded,
        rows_rejected=rows_rejected,
        message=message,
        issues=list(issues),
        run_id=run_id,
    )


def _upsert_match(
    session: Session,
    match: NormalizedMatch,
    *,
    source_id: str,
    import_run_id: int,
) -> None:
    payload = json.dumps(match.raw, ensure_ascii=False, separators=(",", ":"), default=str)
    existing = session.scalar(select(HistoricalMatchRow).where(HistoricalMatchRow.match_key == match.match_key))
    now = datetime.utcnow()
    values = dict(
        source_id=source_id,
        import_run_id=import_run_id,
        match_date=match.match_date,
        tournament=match.tournament,
        round_name=match.round_name,
        surface=match.surface,
        player_a=match.player_a,
        player_b=match.player_b,
        winner=match.winner,
        rank_a=match.rank_a,
        rank_b=match.rank_b,
        points_a=match.points_a,
        points_b=match.points_b,
        best_of=match.best_of,
        tourney_level=match.tourney_level,
        payload_json=payload,
        updated_at=now,
    )
    if existing is None:
        session.add(HistoricalMatchRow(match_key=match.match_key, created_at=now, **values))
        return
    for key, value in values.items():
        setattr(existing, key, value)


def _log_outcome(outcome: ImportOutcome) -> None:
    summary = "import %s source=%s mode=%s rows_seen=%s rows_loaded=%s issues=%s %s"
    args = (
        outcome.status,
        outcome.source_id,
        outcome.mode,
        outcome.rows_seen,
        outcome.rows_loaded,
        len(outcome.issues),
        outcome.message,
    )
    if outcome.status in {"blocked", "error"}:
        logger.error(summary, *args)
    elif outcome.status == "flagged":
        logger.warning(summary, *args)
    else:
        logger.info(summary, *args)
    for issue in outcome.issues:
        level = logging.ERROR if issue.severity == "error" else logging.WARNING
        logger.log(
            level,
            "dq check=%s row=%s field=%s %s",
            issue.check,
            issue.row_index,
            issue.field,
            issue.message,
        )
