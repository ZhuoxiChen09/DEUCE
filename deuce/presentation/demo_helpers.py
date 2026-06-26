"""Deterministic helpers for the portfolio-facing DEUCE demo UI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


FEATURE_LABELS = {
    "rank_diff": "ranking difference",
    "log_points_diff": "ranking-points difference",
    "surface_speed_rating": "surface speed",
    "indoor_flag": "indoor conditions",
    "round_importance": "round context",
    "tournament_weight": "event tier",
    "best_of": "match format",
    "draw_size": "draw size",
    "age_diff": "age difference",
    "height_diff": "height difference",
    "hand_lefty_matchup": "left-handed matchup",
    "seed_diff": "seed difference",
    "seeded_flag_a": "Player A seed status",
    "seeded_flag_b": "Player B seed status",
    "both_seeded_flag": "both players seeded",
    "unseeded_vs_seeded_flag": "seeded-vs-unseeded context",
    "slam_flag": "Grand Slam context",
    "masters_flag": "Masters-level context",
    "atp500_flag": "ATP 500 context",
    "atp250_flag": "ATP 250 context",
    "challenger_flag": "event level",
    "qualifier_flag_a": "Player A qualifier path",
    "qualifier_flag_b": "Player B qualifier path",
    "wildcard_flag_a": "Player A wildcard path",
    "wildcard_flag_b": "Player B wildcard path",
    "same_country_flag": "same-country matchup",
    "home_advantage_diff": "home-country context",
}

GROUP_LABELS = {
    "rank_points": "Ranking and points",
    "tennis_skill": "Serve and return indicators",
    "context": "Match context",
    "interactions": "Feature interactions",
    "recent_form": "Recent form",
    "draw_status": "Draw and seed status",
    "event_level": "Event level",
    "entry_path": "Entry path",
    "event_identity": "Event identity",
    "nationality_home": "Nationality and home context",
    "match_progression": "Match progression",
    "uncertainty_penalty": "Missing-data indicators",
}

FEATURE_MISSINGNESS = {
    "event_id_hash": ("event_id_missing",),
    "both_seeded_flag": ("seeded_flag_a_missing", "seeded_flag_b_missing"),
    "unseeded_vs_seeded_flag": ("seeded_flag_a_missing", "seeded_flag_b_missing"),
    "qualifier_flag_a": ("entry_missing_a",),
    "wildcard_flag_a": ("entry_missing_a",),
    "lucky_loser_flag_a": ("entry_missing_a",),
    "special_exempt_flag_a": ("entry_missing_a",),
    "qualifier_flag_b": ("entry_missing_b",),
    "wildcard_flag_b": ("entry_missing_b",),
    "lucky_loser_flag_b": ("entry_missing_b",),
    "special_exempt_flag_b": ("entry_missing_b",),
    "qualifier_vs_direct_flag": ("entry_missing_a", "entry_missing_b"),
    "wildcard_vs_seeded_flag": (
        "entry_missing_a",
        "entry_missing_b",
        "seeded_flag_a_missing",
        "seeded_flag_b_missing",
    ),
    "same_country_flag": ("home_flag_a_missing", "home_flag_b_missing"),
    "in_event_fatigue_proxy": ("match_num_missing",),
    "int_lefty_surface": ("hand_lefty_missing",),
}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk, returning an empty dict for missing files."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows as dictionaries, returning an empty list when absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def percent(value: Any, *, digits: int = 1) -> str:
    """Format a numeric proportion as a percentage."""
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "Unavailable"


def metric_value(value: Any, *, digits: int = 3) -> str:
    """Format a scalar metric without inventing unavailable values."""
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def confidence_label(probability_a: float) -> tuple[str, str]:
    """
    Label model confidence from the distance between the two probabilities.

    Rules are intentionally transparent:
    - High: leading player is at least 65%.
    - Moderate: leading player is at least 57%.
    - Low: otherwise.
    """
    spread = abs(float(probability_a) - 0.5)
    if spread >= 0.15:
        return "High", "High: leading probability is at least 65%."
    if spread >= 0.07:
        return "Moderate", "Moderate: leading probability is between 57% and 65%."
    return "Low", "Low: probabilities are within seven percentage points of even."


def feature_label(feature_name: str) -> str:
    """Convert a model feature key into a reader-friendly label."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    cleaned = feature_name.replace("_missing", " missing").replace("_", " ")
    return cleaned[:1].upper() + cleaned[1:]


def unavailable_factor_names(feature_values: dict[str, float] | None) -> set[str]:
    """Identify factors whose source values are explicitly marked unavailable."""
    if not feature_values:
        return set()
    unavailable: set[str] = set()
    for feature in contributions_candidate_names(feature_values):
        indicators = FEATURE_MISSINGNESS.get(feature)
        direct_indicator = f"{feature}_missing"
        if indicators is None and direct_indicator in feature_values:
            indicators = (direct_indicator,)
        if indicators and any(
            float(feature_values.get(indicator, 0.0) or 0.0) >= 0.5
            for indicator in indicators
        ):
            unavailable.add(feature)
    return unavailable


def contributions_candidate_names(feature_values: dict[str, float]) -> set[str]:
    """Return non-indicator feature names represented by a feature-value mapping."""
    return {name for name in feature_values if name and not name.endswith("_missing")}


def top_feature_rows(
    contributions: dict[str, float],
    *,
    limit: int = 8,
    feature_values: dict[str, float] | None = None,
    unavailable_features: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return sorted contribution rows for display."""
    unavailable = unavailable_factor_names(feature_values) | (unavailable_features or set())
    rows = [
        {
            "Signal": feature_label(name),
            "Feature": name,
            "Contribution": float(value),
        }
        for name, value in contributions.items()
        if name and name not in unavailable and abs(float(value)) > 1e-9
    ]
    rows.sort(key=lambda row: (-abs(row["Contribution"]), row["Feature"]))
    return rows[:limit]


def grouped_contribution_rows(groups: dict[str, float]) -> list[dict[str, Any]]:
    """Return sorted grouped contribution rows for display."""
    rows = [
        {
            "Group": GROUP_LABELS.get(name, feature_label(name)),
            "Contribution": float(value),
        }
        for name, value in groups.items()
        if name and abs(float(value)) > 1e-9
    ]
    rows.sort(key=lambda row: (-abs(row["Contribution"]), row["Group"]))
    return rows


def deterministic_explanation(
    *,
    player_a: str,
    player_b: str,
    probability_a: float,
    contributions: dict[str, float],
    warnings: list[str] | None = None,
    feature_values: dict[str, float] | None = None,
    unavailable_features: set[str] | None = None,
) -> str:
    """Create a deterministic plain-English prediction explanation."""
    winner = player_a if probability_a >= 0.5 else player_b
    loser = player_b if probability_a >= 0.5 else player_a
    rows = top_feature_rows(
        contributions,
        limit=3,
        feature_values=feature_values,
        unavailable_features=unavailable_features,
    )
    aligned: list[str] = []
    for row in rows:
        contribution = float(row["Contribution"])
        signal = (
            str(row["Signal"])
            .replace("Player A", player_a)
            .replace("Player B", player_b)
        )
        if probability_a >= 0.5 and contribution > 0:
            aligned.append(signal)
        elif probability_a < 0.5 and contribution < 0:
            aligned.append(signal)
    if aligned:
        factors = ", ".join(aligned[:3])
        sentence = f"The model favors {winner} over {loser}, led by {factors}."
    else:
        sentence = f"The model slightly favors {winner}, but the strongest signals are mixed."
    if warnings:
        sentence += " Interpret the result cautiously because some input data is incomplete."
    return sentence


def accuracy_by_confidence_bucket(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compute accuracy by probability-spread buckets from OOS rows."""
    buckets = [
        ("0-5 pp", 0.00, 0.05),
        ("5-10 pp", 0.05, 0.10),
        ("10-15 pp", 0.10, 0.15),
        ("15-20 pp", 0.15, 0.20),
        ("20+ pp", 0.20, 1.00),
    ]
    out: list[dict[str, Any]] = []
    for label, lo, hi in buckets:
        subset = []
        for row in rows:
            try:
                margin = abs(float(row.get("confidence_margin", "")))
                correct = int(float(row.get("correct", "")))
            except (TypeError, ValueError):
                continue
            if lo <= margin < hi:
                subset.append(correct)
        out.append(
            {
                "Confidence bucket": label,
                "Rows": len(subset),
                "Accuracy": (sum(subset) / len(subset)) if subset else None,
            }
        )
    return out
