"""Print / validate feature tier lists for Core, Enhanced, and legacy research."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from deuce.features.leakage import LEAKAGE_RISK_FEATURE_NAMES, assert_no_leakage_in_feature_list
from deuce.features.tier_lists import CORE_FEATURE_NAMES, ENHANCED_FEATURE_NAMES, LEAKAGE_RISK_FEATURES
from model_features import FEATURE_NAMES


def tier_manifest() -> Dict[str, Any]:
    """Structured manifest for CI and documentation."""
    return {
        "core_prematch": {
            "n_features": len(CORE_FEATURE_NAMES),
            "names": CORE_FEATURE_NAMES,
        },
        "enhanced_prematch": {
            "n_features": len(ENHANCED_FEATURE_NAMES),
            "names": ENHANCED_FEATURE_NAMES,
        },
        "legacy_full": {
            "n_features": len(FEATURE_NAMES),
            "names": FEATURE_NAMES,
        },
        "leakage_risk": sorted(LEAKAGE_RISK_FEATURES),
    }


def validate_prematch_tiers() -> None:
    """Raise if Core or Enhanced include leakage-risk names."""
    assert_no_leakage_in_feature_list(CORE_FEATURE_NAMES, context="tier_manifest core")
    assert_no_leakage_in_feature_list(ENHANCED_FEATURE_NAMES, context="tier_manifest enhanced")


def print_feature_tiers_json() -> str:
    validate_prematch_tiers()
    return json.dumps(tier_manifest(), indent=2)


def legacy_includes_leakage() -> List[str]:
    """Names in legacy full vector that are leakage-risk (for documentation)."""
    return [n for n in FEATURE_NAMES if n in LEAKAGE_RISK_FEATURE_NAMES]
