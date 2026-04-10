"""
Evaluation layer on top of the prediction model.
Compares model probability vs market probability, computes edge, confidence, and explanations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Use model_config BUCKET_FEATURES as single source of truth for breakdown (includes base_strength, uncertainty_penalty).
try:
    from model_config import BUCKET_FEATURES
except ImportError:
    from model_features import FEATURE_NAMES
    BUCKET_FEATURES = {
        "base_strength": ["rank_diff", "log_points_diff"],
        "rank_points": ["rank_diff", "log_points_diff"],
        # No-op bucket now: ATP/Elo matchup deltas removed from model features.
        "tennis_skill": [],
        "context": ["surface_speed_rating", "indoor_flag", "altitude_factor", "climate_stress_index", "round_importance", "court_type_indoor", "tournament_weight", "best_of", "draw_size"],
        "match_level": ["age_diff", "height_diff", "hand_lefty_matchup", "ace_rate_diff", "df_rate_diff", "first_serve_in_pct_diff", "first_serve_won_pct_diff", "second_serve_won_pct_diff", "service_games_won_pct_diff", "bp_saved_pct_diff", "bp_faced_pct_diff", "match_minutes", "deciding_set_flag", "int_lefty_surface"],
        "interactions": ["int_serve_surface_speed", "int_return_slow_surface", "int_pressure_round", "int_serve_altitude", "int_return_climate", "int_lefty_surface"],
        "recent_form": ["last5_win_pct_diff", "last10_win_pct_diff", "recent_hold_diff", "recent_break_diff", "deciding_set_diff", "minutes_last_7_days_diff", "days_since_last_match_diff", "timezone_shift_diff", "travel_distance_diff"],
        "draw_status": ["seed_diff", "seeded_flag_a", "seeded_flag_b", "both_seeded_flag", "unseeded_vs_seeded_flag"],
        "event_level": ["slam_flag", "masters_flag", "atp500_flag", "atp250_flag", "challenger_flag"],
        "entry_path": ["qualifier_flag_a", "qualifier_flag_b", "wildcard_flag_a", "wildcard_flag_b", "lucky_loser_flag_a", "lucky_loser_flag_b", "special_exempt_flag_a", "special_exempt_flag_b", "qualifier_vs_direct_flag", "wildcard_vs_seeded_flag"],
        "event_identity": ["event_id_hash"],
        "score_context": ["sets_won_a", "sets_won_b", "straight_sets_flag", "tiebreak_count", "retirement_walkover_flag", "long_match_flag", "comeback_flag"],
        "nationality_home": ["home_flag_a", "home_flag_b", "same_country_flag", "home_advantage_diff"],
        "match_progression": ["match_num", "in_event_fatigue_proxy"],
        "uncertainty_penalty": [n for n in FEATURE_NAMES if n.endswith("_missing")],
    }

@dataclass
class EvaluationResult:
    """Full evaluation: model vs market, edge, confidence, and interpretable breakdown."""
    player_a: str
    player_b: str
    model_prob_a: float
    model_prob_b: float
    market_prob_a: float
    market_prob_b: float
    edge_a: float
    edge_b: float
    confidence_score: str  # "low" | "medium" | "high"
    recommended_side: str  # "A" | "B" | "none"
    feature_breakdown: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def _market_prob_from_price(price: Any) -> float:
    """Convert Kalshi price (e.g. 0.62 for 62 cents) to market probability in [0, 1]."""
    if price is None:
        return 0.5
    try:
        v = float(price)
        if v < 0:
            return 0.5
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.5


def _edge(model_prob: float, market_prob: float) -> float:
    """Edge = model_prob - market_prob (positive = model values this side more than market)."""
    return model_prob - market_prob


def _feature_breakdown_from_values(
    feature_values: Dict[str, float],
    model: Any = None,
    feature_names: Optional[List[str]] = None,
    scaled_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Build grouped feature contribution breakdown from BUCKET_FEATURES.
    Uses scaled_values (post-scaler) when provided so contributions match actual model inputs.
    """
    breakdown: Dict[str, float] = {}
    values_for_contrib = scaled_values if scaled_values is not None else feature_values

    coef_model = model
    if isinstance(model, dict) and model.get("version") in (2, 3):
        coef_model = model.get("model")
    if coef_model is not None and feature_names and hasattr(coef_model, "coef_") and coef_model.coef_ is not None:
        coef = coef_model.coef_[0] if hasattr(coef_model.coef_, "__getitem__") else coef_model.coef_
        try:
            for i, name in enumerate(feature_names):
                if i >= len(coef):
                    break
                val = values_for_contrib.get(name, 0.0)
                try:
                    contrib = float(coef[i]) * float(val)
                except (TypeError, ValueError):
                    contrib = 0.0
                for bucket, keys in BUCKET_FEATURES.items():
                    if name in keys:
                        breakdown[bucket] = breakdown.get(bucket, 0.0) + contrib
        except Exception:
            pass

    return {k: round(v, 3) for k, v in breakdown.items()}


def _explanation_from_breakdown(
    breakdown: Dict[str, float],
    player_a: str,
    player_b: str,
) -> str:
    """Generate short explanation from feature breakdown (positive contributions favor A). Uses bucket names."""
    total = sum(breakdown.values())
    if abs(total) < 0.01:
        return "Model and stats are roughly even between the two players."
    favor_a = total > 0
    winner = player_a if favor_a else player_b
    loser = player_b if favor_a else player_a
    parts: List[str] = []
    for key, val in sorted(breakdown.items(), key=lambda x: -abs(x[1])):
        if abs(val) < 0.005:
            continue
        if key in ("base_strength", "rank_points"):
            parts.append("rank/points")
        elif key == "tennis_skill":
            parts.append("serve/return/pressure")
        elif key == "context":
            parts.append("surface/context")
        elif key == "match_level":
            parts.append("match-level stats")
        elif key == "interactions":
            parts.append("matchup context")
        elif key == "recent_form":
            parts.append("recent form")
        elif key == "draw_status":
            parts.append("draw/seed")
        elif key == "event_level":
            parts.append("event level")
        elif key == "entry_path":
            parts.append("entry path")
        elif key == "event_identity":
            parts.append("event")
        elif key == "score_context":
            parts.append("score context")
        elif key == "nationality_home":
            parts.append("home advantage")
        elif key == "match_progression":
            parts.append("match progression")
        elif key != "uncertainty_penalty":
            parts.append(key.replace("_", " "))
    if not parts:
        return f"Model slightly favors {winner} over {loser}."
    return f"Model favors {winner} due to stronger {', '.join(parts)}."


def _confidence_score(
    feature_values: Dict[str, float],
    edge_a: float,
    edge_b: float,
    model_confidence: str,
) -> str:
    """
    confidence = base + feature_completeness_bonus + edge_strength_bonus.
    Output: "low" | "medium" | "high".
    """
    base = 0.0 if model_confidence == "fallback" else 0.3
    # Feature completeness: count non-zero features
    non_zero = sum(1 for v in feature_values.values() if v is not None and abs(float(v)) > 1e-6)
    total_features = max(len(feature_values), 1)
    completeness = min(1.0, non_zero / max(total_features, 7)) * 0.4
    # Edge strength: larger |edge| => more confidence
    max_edge = max(abs(edge_a), abs(edge_b))
    if max_edge >= 0.15:
        edge_bonus = 0.3
    elif max_edge >= 0.08:
        edge_bonus = 0.2
    elif max_edge >= 0.03:
        edge_bonus = 0.1
    else:
        edge_bonus = 0.0
    score = base + completeness + edge_bonus
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def _recommended_side(edge_a: float, edge_b: float, min_edge: float = 0.02) -> str:
    """Recommend side with positive edge above threshold, else 'none'."""
    if edge_a >= min_edge and edge_a > edge_b:
        return "A"
    if edge_b >= min_edge and edge_b > edge_a:
        return "B"
    return "none"


def evaluate(
    prediction_result: Any,
    market_prob_a: float,
    market_prob_b: Optional[float] = None,
    model: Any = None,
    feature_names: Optional[List[str]] = None,
) -> EvaluationResult:
    """
    Build EvaluationResult from prediction result and market probabilities.
    market_prob_a/b: Prefer **no-vig fair** probabilities in [0,1] when both sides exist;
    else normalized implied. If ``market_prob_b`` omitted, use ``1 - market_prob_a``.
    """
    if market_prob_b is None:
        market_prob_b = 1.0 - market_prob_a
    prob_a = prediction_result.model_prob_a
    prob_b = prediction_result.model_prob_b
    edge_a = _edge(prob_a, market_prob_a)
    edge_b = _edge(prob_b, market_prob_b)

    feature_values = getattr(prediction_result, "feature_values", None) or {}
    scaled_values = getattr(prediction_result, "scaled_feature_values", None)
    breakdown = _feature_breakdown_from_values(
        feature_values, model=model, feature_names=feature_names, scaled_values=scaled_values
    )
    explanation = _explanation_from_breakdown(
        breakdown,
        prediction_result.player_a,
        prediction_result.player_b,
    )
    confidence = _confidence_score(
        feature_values,
        edge_a,
        edge_b,
        getattr(prediction_result, "confidence", "fallback"),
    )
    recommended = _recommended_side(edge_a, edge_b)

    return EvaluationResult(
        player_a=prediction_result.player_a,
        player_b=prediction_result.player_b,
        model_prob_a=prob_a,
        model_prob_b=prob_b,
        market_prob_a=market_prob_a,
        market_prob_b=market_prob_b,
        edge_a=edge_a,
        edge_b=edge_b,
        confidence_score=confidence,
        recommended_side=recommended,
        feature_breakdown=breakdown,
        explanation=explanation,
    )
