from model_features import CORE_FEATURE_NAMES, PlayerStats, build_match_features


def test_build_match_features_core():
    a = PlayerStats(name="A", rank=5.0, points=5000.0)
    b = PlayerStats(name="B", rank=20.0, points=2000.0)
    mf = build_match_features(a, b, feature_set="core")
    assert len(mf.feature_vector) == len(CORE_FEATURE_NAMES)
