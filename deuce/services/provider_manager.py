"""Multi-provider fetch with failover."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from deuce.providers.health import InstrumentedLiveScoreProvider, ProviderHealthTracker
from deuce.providers.interface import LiveScoreProvider

logger = logging.getLogger(__name__)


class OperatingMode(str, Enum):
    """SINGLE: first provider only. FAILOVER: try in order until one succeeds."""

    SINGLE = "single"
    FAILOVER = "failover"


class ProviderManager(LiveScoreProvider):
    """Chains async providers; optional per-attempt health instrumentation."""

    def __init__(
        self,
        providers: list[LiveScoreProvider],
        *,
        mode: OperatingMode = OperatingMode.FAILOVER,
        tracker: ProviderHealthTracker | None = None,
        provider_id: str = "manager",
    ) -> None:
        if not providers:
            raise ValueError("providers must be non-empty")
        self._providers = list(providers)
        self._mode = mode
        self._tracker = tracker
        self._provider_id = provider_id

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _wrap(self, p: LiveScoreProvider) -> LiveScoreProvider:
        if self._tracker is None:
            return p
        return InstrumentedLiveScoreProvider(p, self._tracker)

    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        if self._mode == OperatingMode.SINGLE:
            return await self._wrap(self._providers[0]).fetch_matches(date_strs)

        last_exc: BaseException | None = None
        for p in self._providers:
            try:
                return await self._wrap(p).fetch_matches(date_strs)
            except Exception as exc:
                last_exc = exc
                logger.warning("Provider %s failed: %s", p.provider_id, exc)
                continue
        if last_exc is not None:
            raise last_exc
        return []
