"""Paths, tour identifiers, and environment-style settings."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


project_name = "DEUCE"


class Tour(str, Enum):
    """Tour axis for models, data joins, and segmented metrics."""

    ATP = "ATP"
    WTA = "WTA"
    UNKNOWN = "UNKNOWN"


def default_artifacts_dir(base: Path | None = None) -> Path:
    root = base or Path(__file__).resolve().parents[2]
    return root / "artifacts"
