from deuce.presentation.demo_helpers import (
    accuracy_by_confidence_bucket,
    confidence_label,
    deterministic_explanation,
    top_feature_rows,
)


def test_confidence_label_rules() -> None:
    assert confidence_label(0.66)[0] == "High"
    assert confidence_label(0.58)[0] == "Moderate"
    assert confidence_label(0.53)[0] == "Low"


def test_deterministic_explanation_uses_aligned_signals() -> None:
    text = deterministic_explanation(
        player_a="Player A",
        player_b="Player B",
        probability_a=0.64,
        contributions={"rank_diff": 0.4, "seed_diff": -0.2, "log_points_diff": 0.1},
        warnings=[],
    )

    assert text.startswith("The model favors Player A")
    assert "ranking difference" in text


def test_top_feature_rows_are_sorted_by_absolute_contribution() -> None:
    rows = top_feature_rows({"rank_diff": 0.1, "seed_diff": -0.5, "age_diff": 0.2})

    assert [row["Feature"] for row in rows] == ["seed_diff", "age_diff", "rank_diff"]


def test_missing_seed_features_are_not_displayed() -> None:
    feature_values = {
        "seed_diff": 0.0,
        "seed_diff_missing": 1.0,
        "seeded_flag_a": 0.0,
        "seeded_flag_a_missing": 1.0,
        "seeded_flag_b": 0.0,
        "seeded_flag_b_missing": 1.0,
        "both_seeded_flag": 0.0,
        "unseeded_vs_seeded_flag": 0.0,
        "rank_diff": 3.0,
        "rank_diff_missing": 0.0,
    }
    rows = top_feature_rows(
        {
            "seeded_flag_a": 0.8,
            "both_seeded_flag": -0.7,
            "seed_diff": 0.6,
            "rank_diff": 0.2,
        },
        feature_values=feature_values,
    )

    assert [row["Feature"] for row in rows] == ["rank_diff"]


def test_missing_optional_context_features_are_not_displayed() -> None:
    rows = top_feature_rows(
        {
            "event_id_hash": 0.7,
            "qualifier_flag_a": -0.5,
            "home_advantage_diff": 0.4,
            "log_points_diff": 0.3,
        },
        feature_values={
            "event_id_hash": 0.5,
            "event_id_missing": 1.0,
            "qualifier_flag_a": 0.0,
            "entry_missing_a": 1.0,
            "home_advantage_diff": 0.0,
            "home_advantage_diff_missing": 1.0,
            "log_points_diff": 0.2,
            "log_points_diff_missing": 0.0,
        },
    )

    assert [row["Feature"] for row in rows] == ["log_points_diff"]


def test_populated_inputs_keep_valid_factors() -> None:
    rows = top_feature_rows(
        {"seed_diff": 0.4, "rank_diff": 0.3},
        feature_values={
            "seed_diff": 4.0,
            "seed_diff_missing": 0.0,
            "rank_diff": 2.0,
            "rank_diff_missing": 0.0,
        },
    )

    assert [row["Feature"] for row in rows] == ["seed_diff", "rank_diff"]


def test_contribution_order_is_deterministic_for_ties() -> None:
    rows = top_feature_rows(
        {"seed_diff": -0.5, "age_diff": 0.5, "rank_diff": 0.5}
    )

    assert [row["Feature"] for row in rows] == [
        "age_diff",
        "rank_diff",
        "seed_diff",
    ]


def test_accuracy_by_confidence_bucket() -> None:
    rows = [
        {"confidence_margin": "0.02", "correct": "1"},
        {"confidence_margin": "0.07", "correct": "0"},
        {"confidence_margin": "0.22", "correct": "1"},
    ]

    buckets = accuracy_by_confidence_bucket(rows)

    assert buckets[0]["Rows"] == 1
    assert buckets[1]["Rows"] == 1
    assert buckets[-1]["Accuracy"] == 1.0
