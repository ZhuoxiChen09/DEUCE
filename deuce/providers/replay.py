"""Replay canned scoreboard snapshots (no network)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from deuce.providers.interface import LiveScoreProvider


class ReplayLiveScoreProvider(LiveScoreProvider):
    """Returns successive snapshots per fetch; optionally repeats the last one."""

    def __init__(
        self,
        snapshots: list[list[dict[str, Any]]],
        *,
        provider_id: str = "replay",
        repeat_last: bool = True,
    ) -> None:
        if not snapshots:
            raise ValueError("snapshots must be non-empty")
        self._provider_id = provider_id
        self._snapshots = snapshots
        self._repeat_last = repeat_last
        self._call_idx = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    async def fetch_matches(self, date_strs: list[str]) -> list[dict[str, Any]]:
        _ = date_strs
        if self._call_idx < len(self._snapshots):
            idx = self._call_idx
            self._call_idx += 1
            return copy.deepcopy(self._snapshots[idx])
        if self._repeat_last:
            return copy.deepcopy(self._snapshots[-1])
        return []

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        *,
        provider_id: str = "replay",
        repeat_last: bool = True,
    ) -> ReplayLiveScoreProvider:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "snapshots" in raw:
            frames = raw["snapshots"]
        else:
            frames = raw
        if not isinstance(frames, list):
            raise TypeError("JSON root must be a list or {\"snapshots\": [...]}")
        return cls(frames, provider_id=provider_id, repeat_last=repeat_last)
