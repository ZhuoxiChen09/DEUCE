import json

from deuce.features.tier_manifest import print_feature_tiers_json, validate_prematch_tiers


def test_validate():
    validate_prematch_tiers()


def test_json_roundtrip():
    s = print_feature_tiers_json()
    d = json.loads(s)
    assert "core_prematch" in d
    assert d["core_prematch"]["n_features"] > 0
