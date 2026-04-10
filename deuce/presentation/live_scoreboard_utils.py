"""Pure helpers for compact tennis live-score rendering."""

from __future__ import annotations

from typing import List, Optional, Tuple


def live_score_source_label(source: str) -> str:
    source_key = str(source or "").strip().lower()
    if source_key == "atp-official-public":
        return "ATP Tour live feed"
    if source_key == "wta-official-public":
        return "WTA live feed"
    if source_key == "miami-open-public":
        return "Miami Open live feed"
    if source_key == "flashscore-public-html":
        return "Flashscore match page"
    if source_key == "official":
        return "official live feed"
    if source_key == "espn":
        return "ESPN scoreboard"
    if source_key:
        return source.replace("-", " ")
    return "live scoreboard"


def split_score_tokens(tokens: List[str], *, is_live: bool) -> Tuple[List[str], Optional[str]]:
    vals = [str(t) for t in tokens if str(t).strip()]
    if is_live and len(vals) >= 2 and vals[-1] in {"0", "15", "30", "40"}:
        return vals[:-1], vals[-1]
    return vals, None
