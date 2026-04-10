"""Abstract live score provider interface and sync adapters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from deuce.live_scores import LiveScoreProviderConfig, fetch_scoreboard_by_dates, load_live_score_provider_config


class LiveScoreProvider(ABC):
    """Async source of normalized scoreboard rows (live_scores dict shape)."""

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Stable id for logging, health metrics, and config."""

    @abstractmethod
    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        """Return matches for the given local date keys (YYYYMMDD)."""


SyncFetchFn = Callable[[Sequence[str]], list[dict[str, Any]]]


class SyncFnLiveScoreProvider(LiveScoreProvider):
    """Runs a synchronous fetch function in a thread pool (wraps legacy live_scores)."""

    def __init__(self, provider_id: str, sync_fetch: SyncFetchFn) -> None:
        self._provider_id = provider_id
        self._sync_fetch = sync_fetch

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync_fetch, date_strs)


def default_live_score_provider(
    config: LiveScoreProviderConfig | None = None,
) -> LiveScoreProvider:
    """Provider that mirrors `fetch_scoreboard_by_dates` / `DEUCE_LIVE_SCORE_PROVIDER` env."""
    cfg = config or load_live_score_provider_config()
    sync_fn = partial(fetch_scoreboard_by_dates, config=cfg)
    return SyncFnLiveScoreProvider(f"live_scores:{cfg.provider}", sync_fn)
