"""Paths, tour identifiers, and environment-style settings.

Secrets and deployment settings come from environment variables. The product
database URL is Postgres (``DEUCE_DB_URL``). SQLite is the local and test
default when that variable is unset.
"""

from __future__ import annotations

import os
from datetime import date
from enum import Enum
from pathlib import Path


project_name = "DEUCE"


class Tour(str, Enum):
    """Tour axis for models, data joins, and segmented metrics."""

    ATP = "ATP"
    WTA = "WTA"
    UNKNOWN = "UNKNOWN"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_artifacts_dir(base: Path | None = None) -> Path:
    root = base or repo_root()
    return root / "artifacts"


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def database_url() -> str:
    """Product URL when ``DEUCE_DB_URL`` is set, otherwise a local SQLite file."""
    raw = os.getenv("DEUCE_DB_URL", "").strip()
    if raw:
        return raw
    root = Path(os.getenv("DEUCE_DATA_DIR", ".deuce")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / 'deuce.db'}"


def sqlite_auto_create(url: str | None = None) -> bool:
    """Create tables on startup for SQLite. Postgres uses Alembic."""
    flag = os.getenv("DEUCE_DB_AUTO_CREATE", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag in {"0", "false", "no", "off"}:
        return False
    return (url or database_url()).startswith("sqlite")


def live_scores_enabled() -> bool:
    """Live-score scrapers stay off unless this flag is set."""
    return _flag("DEUCE_LIVE_SCORES_ENABLED", False)


def model_path() -> Path:
    raw = os.getenv("DEUCE_MODEL_PATH", "").strip()
    if raw:
        return resolve_repo_path(raw)
    return repo_root() / "artifacts" / "demo" / "tennis_model_core.pkl"


def player_db_path() -> Path:
    raw = os.getenv("DEUCE_PLAYER_DB", "").strip()
    if raw:
        return resolve_repo_path(raw)
    return repo_root() / "data" / "player_stats.json"


def catalog_path() -> Path:
    raw = os.getenv("DEUCE_TOURNAMENT_CATALOG", "").strip()
    if raw:
        return resolve_repo_path(raw)
    return repo_root() / "data" / "tournament_catalog.json"


def import_enabled() -> bool:
    return _flag("DEUCE_IMPORT_ENABLED", False)


def import_on_start() -> bool:
    return _flag("DEUCE_IMPORT_ON_START", True)


def import_interval_minutes() -> int:
    raw = os.getenv("DEUCE_IMPORT_INTERVAL_MINUTES", "1440").strip() or "1440"
    try:
        value = int(raw)
    except ValueError:
        value = 1440
    return max(1, value)


def import_adapter_name() -> str:
    return os.getenv("DEUCE_IMPORT_ADAPTER", "local_csv").strip() or "local_csv"


def import_source_path() -> Path:
    raw = os.getenv("DEUCE_IMPORT_SOURCE", "data/demo/demo_matches.csv").strip()
    return resolve_repo_path(raw or "data/demo/demo_matches.csv")


def dq_mode() -> str:
    mode = os.getenv("DEUCE_DQ_MODE", "block").strip().lower() or "block"
    if mode not in {"block", "flag"}:
        return "block"
    return mode


def dq_date_bounds() -> tuple[date, date]:
    min_raw = os.getenv("DEUCE_DQ_DATE_MIN", "1990-01-01").strip() or "1990-01-01"
    max_raw = os.getenv("DEUCE_DQ_DATE_MAX", "2035-12-31").strip() or "2035-12-31"
    return date.fromisoformat(min_raw), date.fromisoformat(max_raw)
