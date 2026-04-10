"""
Shared utilities for player name normalization and matching.
Used by the prediction layer to align Kalshi/ATP player names.
"""

import re
import unicodedata
from typing import List, Optional


def normalize_player_name(name: str) -> str:
    """
    Normalize a player name for consistent lookup: lowercase, strip, collapse spaces,
    remove common suffixes (e.g. Jr.), and optional NFKD normalization for accents.
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bjr\.?\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\biii?\s*$", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s


def match_player(
    name: str,
    candidates: List[str],
    *,
    normalize: bool = True,
) -> Optional[str]:
    """
    Find the best matching candidate for the given name.
    Tries exact normalized match first, then first candidate whose normalized form
    contains the name or vice versa.
    """
    if not name or not candidates:
        return None
    needle = normalize_player_name(name) if normalize else name.lower()
    if not needle:
        return None
    normalized_candidates = [(normalize_player_name(c), c) for c in candidates]
    for norm_c, orig in normalized_candidates:
        if norm_c == needle:
            return orig
    for norm_c, orig in normalized_candidates:
        if needle in norm_c or norm_c in needle:
            return orig
    return None
