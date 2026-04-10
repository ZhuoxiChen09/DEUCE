"""Per-provider latency and success tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from deuce.providers.interface import LiveScoreProvider


@dataclass(frozen=True)
class ProviderHealth:
    provider_id: str
    success_count: int = 0
    failure_count: int = 0
    last_latency_ms: float | None = None
    last_error: str | None = None
    last_success_ts: float | None = None
    last_failure_ts: float | None = None


@dataclass
class _MutableStats:
    success_count: int = 0
    failure_count: int = 0
    last_latency_ms: float | None = None
    last_error: str | None = None
    last_success_ts: float | None = None
    last_failure_ts: float | None = None


class ProviderHealthTracker:
    """Thread-safe enough for asyncio single-threaded event loop + thread pool callbacks."""

    def __init__(self) -> None:
        self._stats: dict[str, _MutableStats] = {}

    def _get(self, provider_id: str) -> _MutableStats:
        if provider_id not in self._stats:
            self._stats[provider_id] = _MutableStats()
        return self._stats[provider_id]

    def record_success(self, provider_id: str, latency_ms: float) -> None:
        s = self._get(provider_id)
        s.success_count += 1
        s.last_latency_ms = latency_ms
        s.last_success_ts = time.time()

    def record_failure(self, provider_id: str, latency_ms: float, err: str) -> None:
        s = self._get(provider_id)
        s.failure_count += 1
        s.last_latency_ms = latency_ms
        s.last_error = err
        s.last_failure_ts = time.time()

    def snapshot(self, provider_id: str) -> ProviderHealth:
        s = self._stats.get(provider_id)
        if s is None:
            return ProviderHealth(provider_id=provider_id)
        return ProviderHealth(
            provider_id=provider_id,
            success_count=s.success_count,
            failure_count=s.failure_count,
            last_latency_ms=s.last_latency_ms,
            last_error=s.last_error,
            last_success_ts=s.last_success_ts,
            last_failure_ts=s.last_failure_ts,
        )

    def all_snapshots(self) -> list[ProviderHealth]:
        return [self.snapshot(pid) for pid in sorted(self._stats)]


class InstrumentedLiveScoreProvider(LiveScoreProvider):
    """Wraps a provider and records outcomes on a ProviderHealthTracker."""

    def __init__(self, inner: LiveScoreProvider, tracker: ProviderHealthTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    @property
    def provider_id(self) -> str:
        return self._inner.provider_id

    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        try:
            out = await self._inner.fetch_matches(date_strs)
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000.0
            self._tracker.record_failure(self.provider_id, ms, str(exc))
            raise
        ms = (time.perf_counter() - t0) * 1000.0
        self._tracker.record_success(self.provider_id, ms)
        return out
