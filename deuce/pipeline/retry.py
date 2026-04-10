"""Async retry with exponential backoff."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_s: float = 0.5,
    max_delay_s: float = 8.0,
    jitter: bool = True,
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except BaseException as exc:
            last_exc = exc
            if attempt >= max_attempts - 1:
                raise
            delay = min(base_delay_s * (2**attempt), max_delay_s)
            if jitter:
                delay *= 0.5 + random.random() * 0.5
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
