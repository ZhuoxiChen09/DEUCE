"""
Pre-match vs leakage-risk features.

Production pre-match models must not use target-match score-derived information.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

from deuce.features.tier_lists import LEAKAGE_RISK_FEATURES

# Features that are only valid for post-match / research (legacy training).
LEAKAGE_RISK_FEATURE_NAMES: frozenset[str] = frozenset(LEAKAGE_RISK_FEATURES)


def assert_no_leakage_in_feature_list(names: Sequence[str], *, context: str = "") -> None:
    """Raise ValueError if any leakage-risk name appears in a pre-match feature list."""
    bad = [n for n in names if n in LEAKAGE_RISK_FEATURE_NAMES]
    if bad:
        msg = f"Pre-match feature list contains leakage-risk features: {bad}"
        if context:
            msg = f"{context}: {msg}"
        raise ValueError(msg)


def filter_prematch_names(names: Iterable[str]) -> List[str]:
    """Return names with leakage-risk features removed."""
    return [n for n in names if n not in LEAKAGE_RISK_FEATURE_NAMES]
