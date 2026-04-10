"""Drop duplicate provider updates using score fingerprints."""

from __future__ import annotations

from typing import Any


def score_fingerprint(row: dict[str, Any]) -> str:
    parts: list[str] = [
        str(row.get("status_text") or ""),
        str(row.get("source") or ""),
    ]
    for p in row.get("players") or []:
        if isinstance(p, dict):
            parts.append(str(p.get("name") or ""))
            parts.append(" ".join(str(t) for t in (p.get("score_tokens") or [])))
    return "|".join(parts)


class EventDeduplicator:
    """Memory-backed dedup keyed by (match_key, source)."""

    def __init__(self) -> None:
        self._last: dict[tuple[str, str], str] = {}

    def is_duplicate(self, match_key: str, row: dict[str, Any]) -> bool:
        src = str(row.get("source") or "")
        fp = score_fingerprint(row)
        k = (match_key, src)
        if self._last.get(k) == fp:
            return True
        self._last[k] = fp
        return False

    def reset(self) -> None:
        self._last.clear()
