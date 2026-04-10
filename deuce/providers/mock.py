"""Deterministic canned scoreboard rows for tests and demos."""

from __future__ import annotations

import copy
from typing import Any

from deuce.providers.interface import LiveScoreProvider


class MockLiveScoreProvider(LiveScoreProvider):
    """Returns a fixed list of normalized dicts (same keys as deuce.live_scores)."""

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        *,
        provider_id: str = "mock",
    ) -> None:
        self._provider_id = provider_id
        self._rows = rows if rows is not None else _DEFAULT_ROWS

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        _ = date_strs
        return copy.deepcopy(self._rows)


_DEFAULT_ROWS: list[dict[str, Any]] = [
    {
        "date": "20260323",
        "tournament": "Mock Open",
        "draw": "WTA",
        "status_text": "1st Set",
        "round": "Round 4",
        "court": "Stadium",
        "is_live": True,
        "is_completed": False,
        "is_scheduled": False,
        "players": [
            {"name": "Player A", "score_tokens": ["6", "0", "0"], "country_code": "USA"},
            {"name": "Player B", "score_tokens": ["2", "1", "15"], "country_code": "GBR"},
        ],
        "serve": "2",
        "source": "mock",
        "match_id": "mock-001",
    }
]
