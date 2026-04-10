"""
Match outcome probability prediction.
Loads trained model (bundle: scaler + calibrated LR) or legacy pickle; returns raw + calibrated prob, confidence, completeness, warnings.
Implementation plan: IMPLEMENTATION_PLAN_FEATURES.md. v2 (77) and v3 (127) bundles supported.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

from deuce.training.features import (
    CORE_FEATURE_NAMES,
    ENHANCED_FEATURE_NAMES,
    FEATURE_NAMES,
    CONTEXT_FEATURES,
    INTERACTION_FEATURES,
    RANK_POINTS_FEATURES,
    RECENT_FORM_FEATURES,
    MatchFeatures,
    PlayerStats,
    build_match_features,
    confidence_from_completeness,
    fallback_score,
    feature_dict_from_match_features,
    has_atp_leaderboard_coverage,
    leaderboard_coverage_score,
    should_route_enhanced,
)

try:
    from model_config import (
        BUCKET_FEATURES,
        CONTRIBUTION_ZERO_THRESHOLD,
        ELITE_RANK_THRESHOLD,
        ELITE_SHRINK_STRENGTH,
        EXTREME_PROB_CAP_HI,
        EXTREME_PROB_CAP_LO,
        LOW_COMPLETENESS_THRESHOLD,
        RANK_POINTS_DOMINANCE_THRESHOLD,
    )
except ImportError:
    BUCKET_FEATURES = {
        "rank_points": ["rank_diff", "log_points_diff"],
        "tennis_skill": ["serve_diff", "return_diff", "pressure_diff", "winloss_diff", "tdi_diff"],
        "context": CONTEXT_FEATURES,
        "interactions": INTERACTION_FEATURES,
        "recent_form": RECENT_FORM_FEATURES,
    }
    RANK_POINTS_DOMINANCE_THRESHOLD = 0.5
    ELITE_RANK_THRESHOLD = 15
    ELITE_SHRINK_STRENGTH = 0.3
    LOW_COMPLETENESS_THRESHOLD = 0.7
    EXTREME_PROB_CAP_LO = 0.15
    EXTREME_PROB_CAP_HI = 0.85
    CONTRIBUTION_ZERO_THRESHOLD = 1e-6


@dataclass
class PredictionResult:
    """Result of predicting win probability for player A vs player B."""
    player_a: str
    player_b: str
    model_prob_a: float
    model_prob_b: float
    confidence: str  # "model" or "fallback"
    model_used: str = "legacy"  # "core" | "enhanced" | "legacy" | "fallback"
    evidence_quality: str = "unknown"  # "strong" | "moderate" | "weak"
    feature_values: Dict[str, float] = field(default_factory=dict)
    raw_prob_a: Optional[float] = None
    calibrated_prob_a: Optional[float] = None
    confidence_score: float = 0.0
    data_completeness: float = 0.0
    leaderboard_coverage_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    top_contributions: Dict[str, float] = field(default_factory=dict)
    scaled_feature_values: Optional[Dict[str, float]] = None
    grouped_contributions: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def load_model(model_path: str) -> Tuple[Any, List[str], Dict[str, float]]:
    """
    Load trained model. Returns (bundle_or_model, feature_names, medians).
    If pickle is version-2 bundle: bundle has scaler, calibrated_model, feature_names, medians.
    If legacy: raw classifier, feature_names from .features.json or FEATURE_NAMES, medians = {}.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    import pickle
    import deuce.training.train as _train_mod

    class _BundleUnpickler(pickle.Unpickler):
        """Resolve wrapper classes regardless of whether pkl was produced by model_train or deuce.training.train."""
        def find_class(self, module: str, name: str) -> Any:
            if module in ("__main__", "model_train", "deuce.training.train") and name in ("_PlattCalibratedWrapper", "_BundleCalibratedWrapper"):
                return getattr(_train_mod, name)
            return super().find_class(module, name)

    with open(path, "rb") as f:
        obj = _BundleUnpickler(f).load()

    medians: Dict[str, float] = {}
    if isinstance(obj, dict) and obj.get("version") in (2, 3):
        feature_names = obj.get("feature_names") or FEATURE_NAMES
        medians = obj.get("medians") or {}
        return obj, feature_names, medians

    order_path = path.with_suffix(path.suffix + ".features.json")
    if order_path.exists():
        with open(order_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        feature_names = data.get("feature_names") or data.get("columns") or FEATURE_NAMES
    else:
        feature_names = FEATURE_NAMES
    return obj, feature_names, medians


def _rank_points_contribution(feature_values: Dict[str, float], model: Any, feature_names: List[str]) -> float:
    """Sum of |coef * value| for rank_diff and log_points_diff."""
    if not hasattr(model, "coef_") or model.coef_ is None or not feature_names:
        return 0.0
    coef = model.coef_[0] if hasattr(model.coef_, "__getitem__") else model.coef_
    total = 0.0
    for i, name in enumerate(feature_names):
        if i >= len(coef):
            break
        if name in RANK_POINTS_FEATURES:
            val = feature_values.get(name, 0.0) or 0.0
            total += abs(float(coef[i]) * float(val))
    return total


def _total_contribution(feature_values: Dict[str, float], model: Any, feature_names: List[str]) -> float:
    total = 0.0
    if not hasattr(model, "coef_") or model.coef_ is None or not feature_names:
        return 1.0
    coef = model.coef_[0] if hasattr(model.coef_, "__getitem__") else model.coef_
    for i, name in enumerate(feature_names):
        if i >= len(coef):
            break
        val = feature_values.get(name, 0.0) or 0.0
        total += abs(float(coef[i]) * float(val))
    return total or 1.0


def _top_contributions(feature_values: Dict[str, float], model: Any, feature_names: List[str], top_k: int = 8) -> Dict[str, float]:
    """Use scaled feature_values (post-scaler) so contribution = coef * scaled_value."""
    if not hasattr(model, "coef_") or model.coef_ is None or not feature_names:
        return {}
    coef = model.coef_[0] if hasattr(model.coef_, "__getitem__") else model.coef_
    contribs = []
    for i, name in enumerate(feature_names):
        if i >= len(coef):
            break
        if name.endswith("_missing"):
            continue
        val = feature_values.get(name, 0.0) or 0.0
        c = float(coef[i]) * float(val)
        contribs.append((name, c))
    contribs.sort(key=lambda x: -abs(x[1]))
    return dict(contribs[:top_k])


def _grouped_contributions(
    scaled_feature_values: Dict[str, float],
    model: Any,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute contribution per bucket from scaled values. Includes _missing features in uncertainty_penalty bucket."""
    out: Dict[str, float] = {}
    if not hasattr(model, "coef_") or model.coef_ is None or not feature_names:
        return out
    coef = model.coef_[0] if hasattr(model.coef_, "__getitem__") else model.coef_
    name_to_contrib: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        if i >= len(coef):
            break
        val = scaled_feature_values.get(name, 0.0) or 0.0
        name_to_contrib[name] = float(coef[i]) * float(val)
    for bucket, keys in BUCKET_FEATURES.items():
        total = sum(name_to_contrib.get(k, 0.0) for k in keys if k in name_to_contrib)
        if abs(total) >= CONTRIBUTION_ZERO_THRESHOLD:
            out[bucket] = round(total, 4)
    return out


def predict_match_probability(
    player_a_stats: PlayerStats,
    player_b_stats: PlayerStats,
    model: Any = None,
    model_core: Any = None,
    model_enhanced: Any = None,
    feature_names: Optional[List[str]] = None,
    tournament_name: Optional[str] = None,
    catalog: Optional[Any] = None,
    medians: Optional[Dict[str, float]] = None,
    match_context: Optional[Dict[str, Any]] = None,
) -> PredictionResult:
    """
    Predict P(A wins). Pre-match only (no score-derived leakage in core/enhanced paths).

    Model selection: Enhanced bundle when both players have sufficient ATP/WTA leaderboard
    coverage; otherwise Core when available; else legacy single bundle or fallback heuristic.
    """
    tournament_context = None
    if tournament_name and catalog:
        from deuce.tournament_catalog import get_tournament_context
        tournament_context = get_tournament_context(tournament_name, catalog)

    # Model selection: Enhanced when bundle + coverage score / strict leaderboard stats warrant it.
    cov = should_route_enhanced(player_a_stats, player_b_stats)
    lb_cov_score = leaderboard_coverage_score(player_a_stats, player_b_stats)
    if model is not None and model_core is None and model_enhanced is None:
        active_model = model
        feature_set: Literal["core", "enhanced", "legacy"] = "legacy"
        model_used = "legacy"
    elif model_enhanced is not None and cov:
        active_model = model_enhanced
        feature_set = "enhanced"
        model_used = "enhanced"
    elif model_core is not None:
        active_model = model_core
        feature_set = "core"
        model_used = "core"
    elif model_enhanced is not None:
        active_model = model_enhanced
        feature_set = "enhanced"
        model_used = "enhanced"
    else:
        active_model = model  # may be None
        feature_set = "legacy"
        model_used = "fallback" if model is None else "legacy"

    bundle = isinstance(active_model, dict) and active_model.get("version") in (2, 3)
    if bundle and medians is None:
        medians = active_model.get("medians") or {}
    mf = build_match_features(
        player_a_stats, player_b_stats,
        tournament_context=tournament_context,
        medians=medians,
        match_context=match_context,
        feature_set=feature_set,
    )
    feature_values = feature_dict_from_match_features(mf)
    names = feature_names or (active_model.get("feature_names") if bundle else (CORE_FEATURE_NAMES if feature_set == "core" else ENHANCED_FEATURE_NAMES if feature_set == "enhanced" else FEATURE_NAMES))

    prob_a: float
    raw_a: Optional[float] = None
    calibrated_a: Optional[float] = None
    confidence_str = "fallback"
    warnings_list: List[str] = []
    if model_enhanced is not None and not cov and model_core is not None and feature_set == "core":
        warnings_list.append("Routed to Core: insufficient ATP/WTA leaderboard coverage for Enhanced.")
    completeness = confidence_from_completeness(mf.n_missing, mf.n_total)
    top_contrib = {}
    evidence_quality = "weak" if completeness < 0.5 else ("strong" if completeness >= 0.8 else "moderate")

    predictor = active_model.get("calibrated_model") if bundle else active_model
    scaler = active_model.get("scaler") if bundle else None

    scaled_feature_values: Optional[Dict[str, float]] = None
    grouped_contrib: Dict[str, float] = {}
    bundle_version = active_model.get("version") if bundle else None
    n_bundle = len(names) if names else 0
    n_builder = len(mf.feature_vector)
    diagnostics: Dict[str, Any] = {
        "leaderboard_coverage_score": lb_cov_score,
        "missing_count": mf.n_missing,
        "total_features": mf.n_total,
        "bundle_version": bundle_version,
        "feature_count_in_bundle": n_bundle,
        "feature_count_from_builder": n_builder,
        "feature_count_mismatch": (n_builder != n_bundle) if names else None,
    }
    if bundle_version is not None:
        logger.debug("Loaded bundle v%s (%s features); builder produced %s features.", bundle_version, n_bundle, n_builder)
    if names and n_builder != n_bundle:
        logger.warning("Feature count mismatch: bundle has %s, builder produced %s.", n_bundle, n_builder)

    def _swap_match_context(mc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not mc:
            return mc
        out = dict(mc)
        a_keys = [k for k in out.keys() if k.endswith("_a")]
        for k_a in a_keys:
            k_b = k_a[:-2] + "_b"
            if k_b in out:
                out[k_a], out[k_b] = out[k_b], out[k_a]
        return out

    def _predict_prob_a_for(
        pa: PlayerStats,
        pb: PlayerStats,
        mc_in: Optional[Dict[str, Any]],
        *,
        feature_set_in: Literal["core", "enhanced", "legacy"],
    ) -> Optional[float]:
        if active_model is None or predictor is None or not names:
            return None
        try:
            import numpy as np
            mf_local = build_match_features(
                pa, pb,
                tournament_context=tournament_context,
                medians=medians,
                match_context=mc_in,
                feature_set=feature_set_in,
            )
            fv_local = feature_dict_from_match_features(mf_local)
            vec_local = [fv_local.get(n, 0.0) for n in names]
            X_local = np.array([vec_local], dtype=float)
            if scaler is not None:
                X_local = scaler.transform(X_local)
            if hasattr(predictor, "predict_proba"):
                probs_local = predictor.predict_proba(X_local)[0]
                idx_local = 1 if (getattr(predictor, "classes_", None) is not None and list(predictor.classes_) == [0, 1]) else 0
                return float(probs_local[idx_local])
        except Exception:
            return None
        return None

    if active_model is not None and predictor is not None and names and len(mf.feature_vector) >= len(names):
        try:
            import numpy as np
            vec = [feature_values.get(n, 0.0) for n in names]
            X = np.array([vec], dtype=float)
            if scaler is not None:
                X = scaler.transform(X)
            scaled_feature_values = {names[i]: float(X[0, i]) for i in range(len(names))}

            if hasattr(predictor, "predict_proba"):
                probs = predictor.predict_proba(X)[0]
                idx_a = 1 if (getattr(predictor, "classes_", None) is not None and list(predictor.classes_) == [0, 1]) else 0
                calibrated_a = float(probs[idx_a])
            else:
                calibrated_a = 0.5
            raw_model = active_model.get("model") if bundle else predictor
            if raw_model is not None and hasattr(raw_model, "predict_proba"):
                raw_probs = raw_model.predict_proba(X)[0]
                idx_a = 1 if list(getattr(raw_model, "classes_", [0, 1])) == [0, 1] else 0
                raw_a = float(raw_probs[idx_a])
            else:
                raw_a = calibrated_a
            prob_a = calibrated_a
            confidence_str = "model"

            inner = active_model.get("model") if bundle else predictor
            top_contrib = _top_contributions(scaled_feature_values, inner, names)
            rank_contrib = _rank_points_contribution(scaled_feature_values, inner, names)
            total_contrib = _total_contribution(scaled_feature_values, inner, names)
            grouped_contrib = _grouped_contributions(scaled_feature_values, inner, names)

            # --- Symmetry enforcement (player-order bias guard) ---
            # For a well-formed winner model, swapping A/B should approximately flip probabilities:
            #   p(A beats B) ≈ 1 - p(B beats A).
            # In practice, slot bias can appear from training row ordering or symmetric-only features.
            # We enforce symmetry at inference by averaging:
            #   p_sym = 0.5 * (p + (1 - p_swap))
            if feature_set in ("core", "enhanced"):
                p_swap = _predict_prob_a_for(
                    player_b_stats,
                    player_a_stats,
                    _swap_match_context(match_context),
                    feature_set_in=feature_set,
                )
                if p_swap is not None:
                    symmetry_error = abs(prob_a - (1.0 - p_swap))
                    diagnostics["symmetry_p_swap_a_wins"] = p_swap
                    diagnostics["symmetry_error_abs"] = symmetry_error
                    prob_a = 0.5 * (prob_a + (1.0 - p_swap))
                    calibrated_a = prob_a
                    if symmetry_error >= 0.10:
                        warnings_list.append(
                            f"Player-order symmetry check failed (|p - (1-p_swap)|={symmetry_error:.3f}); "
                            "probability was symmetrized to reduce slot bias."
                        )

            # Single-bucket dominance: shrink extreme probs when one bucket dominates
            if total_contrib > 0:
                max_bucket = max(abs(c) for c in grouped_contrib.values()) if grouped_contrib else 0
                if max_bucket / total_contrib >= 0.5:
                    prob_a = prob_a * 0.7 + 0.5 * 0.3
                    calibrated_a = prob_a
                    warnings_list.append("Prediction heavily driven by one factor; probability pulled toward 50%.")
            if total_contrib > 0 and rank_contrib / total_contrib >= RANK_POINTS_DOMINANCE_THRESHOLD:
                warnings_list.append("Prediction relies heavily on ranking prior.")
            if completeness < 0.5:
                warnings_list.append("Many features missing; confidence reduced.")
            if getattr(player_a_stats, "serve_rating", None) is None and getattr(player_b_stats, "serve_rating", None) is None:
                warnings_list.append("Limited surface-specific evidence (serve/return data).")
            if completeness < LOW_COMPLETENESS_THRESHOLD and (prob_a >= 0.9 or prob_a <= 0.1):
                prob_a = max(EXTREME_PROB_CAP_LO, min(EXTREME_PROB_CAP_HI, prob_a))
                calibrated_a = prob_a
                warnings_list.append("Extreme probability capped due to low data quality.")

            # Elite-vs-elite: shrink toward 0.5
            ra, rb = getattr(player_a_stats, "rank", None), getattr(player_b_stats, "rank", None)
            if ra is not None and rb is not None:
                try:
                    if int(ra) <= ELITE_RANK_THRESHOLD and int(rb) <= ELITE_RANK_THRESHOLD:
                        prob_a = prob_a * (1 - ELITE_SHRINK_STRENGTH) + 0.5 * ELITE_SHRINK_STRENGTH
                        calibrated_a = prob_a
                        warnings_list.append("Elite vs elite: probability pulled toward 50%.")
                except (TypeError, ValueError):
                    pass

            diagnostics["raw_feature_values"] = dict(feature_values)
            diagnostics["scaled_feature_values"] = dict(scaled_feature_values)
            coef_row = inner.coef_[0] if hasattr(inner.coef_, "__getitem__") else inner.coef_
            diagnostics["per_feature_contribution"] = {n: round(float(coef_row[i]) * float(scaled_feature_values.get(n, 0)), 4) for i, n in enumerate(names) if i < len(coef_row) and not n.endswith("_missing")}
            diagnostics["grouped_contributions"] = dict(grouped_contrib)
        except Exception:
            prob_a = fallback_score(player_a_stats, player_b_stats)
            raw_a = calibrated_a = prob_a
            confidence_str = "fallback"
            model_used = "fallback"
    else:
        prob_a = fallback_score(player_a_stats, player_b_stats)
        raw_a = calibrated_a = prob_a
        confidence_str = "fallback"
        model_used = "fallback"

    prob_a = max(0.0, min(1.0, prob_a))
    prob_b = 1.0 - prob_a
    diagnostics["model_used"] = model_used
    diagnostics["evidence_quality"] = evidence_quality
    return PredictionResult(
        player_a=player_a_stats.name,
        player_b=player_b_stats.name,
        model_prob_a=prob_a,
        model_prob_b=prob_b,
        confidence=confidence_str,
        model_used=model_used,
        evidence_quality=evidence_quality,
        feature_values=feature_values,
        raw_prob_a=raw_a,
        calibrated_prob_a=calibrated_a,
        confidence_score=completeness,
        data_completeness=completeness,
        leaderboard_coverage_score=lb_cov_score,
        warnings=warnings_list,
        top_contributions=top_contrib,
        scaled_feature_values=scaled_feature_values,
        grouped_contributions=grouped_contrib,
        diagnostics=diagnostics,
    )


def get_default_model_path() -> Optional[str]:
    """Return default legacy model path if artifacts/tennis_model.pkl exists."""
    candidates = [
        Path(__file__).resolve().parent / "artifacts" / "tennis_model.pkl",
        Path.cwd() / "artifacts" / "tennis_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def get_default_core_path() -> Optional[str]:
    """Return path to Core pre-match model (artifacts/tennis_model_core.pkl)."""
    candidates = [
        Path(__file__).resolve().parent / "artifacts" / "tennis_model_core.pkl",
        Path.cwd() / "artifacts" / "tennis_model_core.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def get_default_enhanced_path() -> Optional[str]:
    """Return path to Enhanced model (artifacts/tennis_model_enhanced.pkl)."""
    candidates = [
        Path(__file__).resolve().parent / "artifacts" / "tennis_model_enhanced.pkl",
        Path.cwd() / "artifacts" / "tennis_model_enhanced.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None
