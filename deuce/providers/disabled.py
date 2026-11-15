"""Live-score provider that performs no network I/O.

The product API uses this unless ``DEUCE_LIVE_SCORES_ENABLED`` is set.
"""

from __future__ import annotations

from typing import Any

from deuce.providers.interface import LiveScoreProvider


class DisabledLiveScoreProvider(LiveScoreProvider):
    """Returns no matches. Scrapers stay off on the product path."""

    @property
    def provider_id(self) -> str:
        return "disabled"

    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        _ = date_strs
        return []
