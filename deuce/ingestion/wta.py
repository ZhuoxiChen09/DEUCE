"""
WTA ingestion placeholder.

Future: parallel pipelines to ATP (rankings, optional stats) keyed by ``Tour.WTA``.
"""

from __future__ import annotations

from typing import Protocol


class WtaStatsIngestion(Protocol):
    """Interface for future WTA leaderboard / ranking ingestion."""

    def fetch_rankings(self, limit: int = 200) -> list[dict]:
        ...


def wta_not_implemented_hint() -> str:
    return "WTA-specific ingestion is not implemented; use tour=WTA in DB and train WTA bundles when data is available."
