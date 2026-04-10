import pytest

from deuce.features.leakage import LEAKAGE_RISK_FEATURE_NAMES, assert_no_leakage_in_feature_list, filter_prematch_names
from model_features import CORE_FEATURE_NAMES, ENHANCED_FEATURE_NAMES


def test_core_and_enhanced_have_no_leakage_columns():
    assert_no_leakage_in_feature_list(CORE_FEATURE_NAMES)
    assert_no_leakage_in_feature_list(ENHANCED_FEATURE_NAMES)


def test_leakage_set_nonempty():
    assert "sets_won_a" in LEAKAGE_RISK_FEATURE_NAMES


def test_filter_prematch_names():
    names = ["rank_diff", "sets_won_a", "log_points_diff"]
    assert filter_prematch_names(names) == ["rank_diff", "log_points_diff"]


def test_assert_raises_on_leakage():
    with pytest.raises(ValueError):
        assert_no_leakage_in_feature_list(["sets_won_a"])
