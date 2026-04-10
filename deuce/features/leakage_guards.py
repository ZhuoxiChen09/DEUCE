"""Re-exports for CI / tooling (single import path for leakage checks)."""

from deuce.features.leakage import LEAKAGE_RISK_FEATURE_NAMES, assert_no_leakage_in_feature_list

__all__ = ["LEAKAGE_RISK_FEATURE_NAMES", "assert_no_leakage_in_feature_list"]
