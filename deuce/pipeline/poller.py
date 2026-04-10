"""Async polling loop around a LiveScoreProvider."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from deuce.pipeline.retry import retry_async
from deuce.providers.interface import LiveScoreProvider

logger = logging.getLogger(__name__)

FetchHandler = Callable[[list[dict[str, Any]]], Awaitable[None]]


class LiveScorePoller:
    def __init__(
        self,
        provider: LiveScoreProvider,
        *,
        interval_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._provider = provider
        self._interval_s = interval_s
        self._max_retries = max_retries
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def poll_once(self, date_strs: list[str]) -> list[dict[str, Any]]:
        async def _call():
            return await self._provider.fetch_matches(date_strs)

        return await retry_async(_call, max_attempts=self._max_retries)

    async def run(
        self,
        date_strs: list[str],
        on_matches: FetchHandler,
    ) -> None:
        while not self._stop.is_set():
            try:
                rows = await self.poll_once(date_strs)
                await on_matches(rows)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("LiveScorePoller tick failed for %s", date_strs)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
                break
            except asyncio.TimeoutError:
                continue
