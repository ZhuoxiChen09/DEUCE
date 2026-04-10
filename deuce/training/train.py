"""
Train a logistic regression classifier on historical tennis match data.
Reads CSV, engineers features (same as model_features), trains, saves model + feature order.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps for training
try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, accuracy_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    np = None  # type: ignore
    LogisticRegression = None  # type: ignore
    train_test_split = None
    log_loss = None
    accuracy_score = None
    StandardScaler = None  # type: ignore
    _import_err = e


def _fit_platt_calibration(model: Any, X_val: Any, y_val: Any) -> Any:
    """Platt scaling: fit a LogisticRegression on (model.predict_proba(X_val)[:, 1], y_val). Works on any sklearn version."""
    raw_prob = model.predict_proba(X_val)[:, 1]
    # Clip to avoid log(0) in logistic regression
    raw_prob = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
    platt = LogisticRegression(max_iter=500, random_state=42)
    platt.fit(raw_prob.reshape(-1, 1), y_val)
    return platt


class _PlattCalibratedWrapper:
    """Wrapper that applies Platt scaling to base model predictions. Same interface as a classifier for predict_proba/predict."""

    def __init__(self, base_model: Any, platt: Any):
        self.base_model = base_model
        self.platt = platt
        self.classes_ = getattr(platt, "classes_", np.array([0, 1]))

    def predict_proba(self, X: Any) -> Any:
        raw = self.base_model.predict_proba(X)[:, 1]
        raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
        return self.platt.predict_proba(raw.reshape(-1, 1))

    def predict(self, X: Any) -> Any:
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)


class _BundleCalibratedWrapper:
    """Calibration via CalibratorBundle (none / platt / isotonic), fit on validation only."""

    def __init__(self, base_model: Any, bundle: Any):
        self.base_model = base_model
        self.calibrator_bundle = bundle
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: Any) -> Any:
        raw = self.base_model.predict_proba(X)[:, 1]
        if apply_calibration is None or CalibratorBundle is None:
            cal = raw
        else:
            cal = apply_calibration(self.calibrator_bundle, raw)
        cal = np.clip(np.asarray(cal, dtype=float), 1e-9, 1.0 - 1e-9)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: Any) -> Any:
        p = self.predict_proba(X)[:, 1]
        return (p >= 0.5).astype(int)

from deuce.training.features import (
    CORE_FEATURE_NAMES,
    DEFAULT_MEDIANS,
    ENHANCED_FEATURE_NAMES,
    FEATURE_NAMES,
    PlayerStats,
    build_match_features,
    parse_score,
    safe_float,
)

try:
    from deuce.features.leakage import assert_no_leakage_in_feature_list
except ImportError:
    def assert_no_leakage_in_feature_list(names, *, context: str = "") -> None:  # type: ignore
        pass

try:
    from deuce.training.calibration import (
        CalibratorBundle,
        CalibrationMethod,
        apply_calibration,
        fit_calibration,
    )
    from deuce.training.splits import SplitStrategy, parse_match_date_int, split_train_val_indices
except ImportError:
    CalibratorBundle = None  # type: ignore
    CalibrationMethod = str  # type: ignore
    apply_calibration = None  # type: ignore
    fit_calibration = None  # type: ignore
    parse_match_date_int = None  # type: ignore
    split_train_val_indices = None  # type: ignore
    SplitStrategy = str  # type: ignore

CODE_VERSION = "deuce-0.3.0"


def _parse_row_date(row: Dict[str, Any]) -> int:
    if parse_match_date_int is None:
        return 0
    return int(parse_match_date_int(row))


def _missingness_summary(X: Any, feature_names: List[str]) -> Dict[str, float]:
    """Fraction of rows where each *_missing indicator is active (train set)."""
    summary: Dict[str, float] = {}
    for i, name in enumerate(feature_names):
        if i >= X.shape[1]:
            break
        if name.endswith("_missing"):
            summary[name] = float(np.mean(X[:, i] > 0.5))
    return summary


def _date_window_str(dates: np.ndarray, idx: np.ndarray) -> Dict[str, Optional[int]]:
    if idx.size == 0:
        return {"min": None, "max": None}
    d = dates[idx]
    d = d[d > 0]
    if d.size == 0:
        return {"min": None, "max": None}
    return {"min": int(np.min(d)), "max": int(np.max(d))}


def fit_calibrated_lr_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray],
    *,
    calibration_method: str = "platt",
    random_state: int = 42,
    sample_weight_train: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit StandardScaler + LogisticRegression on train; calibrate **only** on validation rows;
    return calibrated probabilities on test. Used by walk-forward and tests.

    Does not persist to disk. When ``X_test`` is empty, ``prob_a_test`` is empty.
    """
    if np is None or LogisticRegression is None or StandardScaler is None:
        raise RuntimeError("Training requires numpy and scikit-learn.")

    cal_m = (calibration_method or "platt").lower()
    if cal_m not in ("none", "platt", "isotonic"):
        cal_m = "platt"

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_s, y_train, sample_weight=sample_weight_train)

    raw_val = model.predict_proba(X_val_s)[:, 1]
    raw_ll = log_loss(y_val, np.clip(raw_val, 1e-15, 1.0 - 1e-15))

    if fit_calibration is None or CalibratorBundle is None:
        platt = _fit_platt_calibration(model, X_val_s, y_val)
        calibrated: Any = _PlattCalibratedWrapper(model, platt)
        cal_m = "platt"
    elif cal_m == "none":
        cal_bundle = CalibratorBundle(method="none")
        calibrated = _BundleCalibratedWrapper(model, cal_bundle)
    else:
        cal_bundle = fit_calibration(cal_m, raw_val, y_val, random_state=random_state)  # type: ignore[operator]
        calibrated = _BundleCalibratedWrapper(model, cal_bundle)

    prob_a_val = calibrated.predict_proba(X_val_s)[:, 1]
    y_val_pred = calibrated.predict(X_val_s)
    ll_cal = log_loss(y_val, prob_a_val)
    acc = accuracy_score(y_val, y_val_pred)

    prob_a_test = np.array([], dtype=float)
    test_ll: Optional[float] = None
    test_brier: Optional[float] = None
    if X_test.shape[0] > 0:
        X_test_s = scaler.transform(X_test)
        prob_a_test = calibrated.predict_proba(X_test_s)[:, 1]
        if y_test is not None and len(y_test) == len(prob_a_test):
            test_ll = float(log_loss(y_test, np.clip(prob_a_test, 1e-15, 1.0 - 1e-15)))
            test_brier = float(np.mean((prob_a_test - y_test.astype(float)) ** 2))

    return {
        "scaler": scaler,
        "model": model,
        "calibrated_model": calibrated,
        "calibration_method": cal_m,
        "val_log_loss_raw": float(raw_ll),
        "val_log_loss_calibrated": float(ll_cal),
        "val_accuracy": float(acc),
        "prob_a_test": prob_a_test,
        "test_log_loss": test_ll,
        "test_brier": test_brier,
        "calibration": {
            "method": cal_m,
            "val_log_loss_raw": float(raw_ll),
            "val_log_loss_calibrated": float(ll_cal),
            "val_accuracy": float(acc),
        },
    }


def _match_context_from_row(row: Dict[str, Any], swap: bool = False) -> Dict[str, Any]:
    """Build match_context dict from a CSV row (historical or ATP/TML). If swap, row has winner/loser; map to player_a/player_b."""
    mc: Dict[str, Any] = {}
    if swap:
        mc["hand_a"] = (row.get("loser_hand") or "").strip()
        mc["hand_b"] = (row.get("winner_hand") or "").strip()
        mc["height_a"] = safe_float(row.get("loser_ht"))
        mc["height_b"] = safe_float(row.get("winner_ht"))
        mc["age_a"] = safe_float(row.get("loser_age"))
        mc["age_b"] = safe_float(row.get("winner_age"))
        mc["ace_a"] = safe_float(row.get("l_ace"))
        mc["ace_b"] = safe_float(row.get("w_ace"))
        mc["df_a"] = safe_float(row.get("l_df"))
        mc["df_b"] = safe_float(row.get("w_df"))
        mc["svpt_a"] = safe_float(row.get("l_svpt"))
        mc["svpt_b"] = safe_float(row.get("w_svpt"))
        mc["1stIn_a"] = safe_float(row.get("l_1stIn"))
        mc["1stIn_b"] = safe_float(row.get("w_1stIn"))
        mc["1stWon_a"] = safe_float(row.get("l_1stWon"))
        mc["1stWon_b"] = safe_float(row.get("w_1stWon"))
        mc["2ndWon_a"] = safe_float(row.get("l_2ndWon"))
        mc["2ndWon_b"] = safe_float(row.get("w_2ndWon"))
        mc["SvGms_a"] = safe_float(row.get("l_SvGms"))
        mc["SvGms_b"] = safe_float(row.get("w_SvGms"))
        mc["bpSaved_a"] = safe_float(row.get("l_bpSaved"))
        mc["bpSaved_b"] = safe_float(row.get("w_bpSaved"))
        mc["bpFaced_a"] = safe_float(row.get("l_bpFaced"))
        mc["bpFaced_b"] = safe_float(row.get("w_bpFaced"))
    else:
        mc["hand_a"] = (row.get("winner_hand") or "").strip()
        mc["hand_b"] = (row.get("loser_hand") or "").strip()
        mc["height_a"] = safe_float(row.get("winner_ht"))
        mc["height_b"] = safe_float(row.get("loser_ht"))
        mc["age_a"] = safe_float(row.get("winner_age"))
        mc["age_b"] = safe_float(row.get("loser_age"))
        mc["ace_a"] = safe_float(row.get("w_ace"))
        mc["ace_b"] = safe_float(row.get("l_ace"))
        mc["df_a"] = safe_float(row.get("w_df"))
        mc["df_b"] = safe_float(row.get("l_df"))
        mc["svpt_a"] = safe_float(row.get("w_svpt"))
        mc["svpt_b"] = safe_float(row.get("l_svpt"))
        mc["1stIn_a"] = safe_float(row.get("w_1stIn"))
        mc["1stIn_b"] = safe_float(row.get("l_1stIn"))
        mc["1stWon_a"] = safe_float(row.get("w_1stWon"))
        mc["1stWon_b"] = safe_float(row.get("l_1stWon"))
        mc["2ndWon_a"] = safe_float(row.get("w_2ndWon"))
        mc["2ndWon_b"] = safe_float(row.get("l_2ndWon"))
        mc["SvGms_a"] = safe_float(row.get("w_SvGms"))
        mc["SvGms_b"] = safe_float(row.get("l_SvGms"))
        mc["bpSaved_a"] = safe_float(row.get("w_bpSaved"))
        mc["bpSaved_b"] = safe_float(row.get("l_bpSaved"))
        mc["bpFaced_a"] = safe_float(row.get("w_bpFaced"))
        mc["bpFaced_b"] = safe_float(row.get("l_bpFaced"))
    mc["best_of"] = safe_float(row.get("best_of"))
    mc["draw_size"] = safe_float(row.get("draw_size"))
    mc["minutes"] = safe_float(row.get("minutes"))
    dec = row.get("deciding_set")
    if dec is not None and str(dec).strip() in ("1", "yes", "true"):
        mc["deciding_set"] = 1
    if swap:
        mc["seed_a"] = safe_float(row.get("loser_seed"))
        mc["seed_b"] = safe_float(row.get("winner_seed"))
        mc["entry_a"] = (row.get("loser_entry") or "").strip()
        mc["entry_b"] = (row.get("winner_entry") or "").strip()
        mc["ioc_a"] = (row.get("loser_ioc") or "").strip()
        mc["ioc_b"] = (row.get("winner_ioc") or "").strip()
    else:
        mc["seed_a"] = safe_float(row.get("winner_seed"))
        mc["seed_b"] = safe_float(row.get("loser_seed"))
        mc["entry_a"] = (row.get("winner_entry") or "").strip()
        mc["entry_b"] = (row.get("loser_entry") or "").strip()
        mc["ioc_a"] = (row.get("winner_ioc") or "").strip()
        mc["ioc_b"] = (row.get("loser_ioc") or "").strip()
    mc["tourney_id"] = (row.get("tourney_id") or "").strip()
    mc["tourney_level"] = (row.get("tourney_level") or "").strip()
    mc["match_num"] = safe_float(row.get("match_num"))
    mc["score"] = (row.get("score") or "").strip()
    parsed = parse_score(mc.get("score"))
    mc["score_parsed"] = parsed
    if parsed.get("sets_winner") is not None and parsed.get("sets_loser") is not None:
        if swap:
            mc["sets_won_a"] = float(parsed["sets_loser"])
            mc["sets_won_b"] = float(parsed["sets_winner"])
        else:
            mc["sets_won_a"] = float(parsed["sets_winner"])
            mc["sets_won_b"] = float(parsed["sets_loser"])
    return mc


def _match_context_from_historical_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build match_context from historical CSV row (columns hand_a, hand_b, ht_a, age_a, ace_a, ...)."""
    mc: Dict[str, Any] = {}
    mc["hand_a"] = (row.get("hand_a") or "").strip()
    mc["hand_b"] = (row.get("hand_b") or "").strip()
    mc["height_a"] = safe_float(row.get("ht_a"))
    mc["height_b"] = safe_float(row.get("ht_b"))
    mc["age_a"] = safe_float(row.get("age_a"))
    mc["age_b"] = safe_float(row.get("age_b"))
    mc["ace_a"] = safe_float(row.get("ace_a"))
    mc["ace_b"] = safe_float(row.get("ace_b"))
    mc["df_a"] = safe_float(row.get("df_a"))
    mc["df_b"] = safe_float(row.get("df_b"))
    mc["svpt_a"] = safe_float(row.get("svpt_a"))
    mc["svpt_b"] = safe_float(row.get("svpt_b"))
    mc["1stIn_a"] = safe_float(row.get("1stIn_a"))
    mc["1stIn_b"] = safe_float(row.get("1stIn_b"))
    mc["1stWon_a"] = safe_float(row.get("1stWon_a"))
    mc["1stWon_b"] = safe_float(row.get("1stWon_b"))
    mc["2ndWon_a"] = safe_float(row.get("2ndWon_a"))
    mc["2ndWon_b"] = safe_float(row.get("2ndWon_b"))
    mc["SvGms_a"] = safe_float(row.get("SvGms_a"))
    mc["SvGms_b"] = safe_float(row.get("SvGms_b"))
    mc["bpSaved_a"] = safe_float(row.get("bpSaved_a"))
    mc["bpSaved_b"] = safe_float(row.get("bpSaved_b"))
    mc["bpFaced_a"] = safe_float(row.get("bpFaced_a"))
    mc["bpFaced_b"] = safe_float(row.get("bpFaced_b"))
    mc["best_of"] = safe_float(row.get("best_of"))
    mc["draw_size"] = safe_float(row.get("draw_size"))
    mc["minutes"] = safe_float(row.get("minutes"))
    dec = row.get("deciding_set")
    if dec is not None and str(dec).strip() in ("1", "yes", "true"):
        mc["deciding_set"] = 1
    mc["seed_a"] = safe_float(row.get("seed_a"))
    mc["seed_b"] = safe_float(row.get("seed_b"))
    mc["entry_a"] = (row.get("entry_a") or "").strip()
    mc["entry_b"] = (row.get("entry_b") or "").strip()
    mc["tourney_id"] = (row.get("tourney_id") or "").strip()
    mc["tourney_level"] = (row.get("tourney_level") or "").strip()
    mc["ioc_a"] = (row.get("ioc_a") or "").strip()
    mc["ioc_b"] = (row.get("ioc_b") or "").strip()
    mc["match_num"] = safe_float(row.get("match_num"))
    mc["score"] = (row.get("score") or "").strip()
    mc["sets_won_a"] = safe_float(row.get("sets_won_a"))
    mc["sets_won_b"] = safe_float(row.get("sets_won_b"))
    mc["score_parsed"] = parse_score(mc.get("score"))
    return mc


def _surface_from_atp(surface: str, indoor: str) -> str:
    """Map ATP surface + indoor flag to catalog surface (clay, grass, hard_outdoor, hard_indoor)."""
    s = (surface or "").strip().lower()
    ind = (indoor or "").strip().upper() == "I"
    if "clay" in s or s == "clay":
        return "clay"
    if "grass" in s or s == "grass":
        return "grass"
    if "hard" in s or s == "hard":
        return "hard_indoor" if ind else "hard_outdoor"
    return "hard_outdoor"


def load_atp_match_csv(
    path: str,
    catalog: Optional[Any] = None,
    *,
    seed: Optional[int] = 42,
    feature_set: str = "core",
) -> Tuple[List[Dict[str, Any]], List[int], Optional[List[str]]]:
    """
    Load ATP-format CSV (winner_name, loser_name, winner_rank, loser_rank, winner_rank_points,
    loser_rank_points, tourney_name, surface, indoor, tourney_date).
    Randomly swap winner/loser per row so we get both classes (A wins / B wins) for training.
    """
    import csv
    import random
    from deuce.tournament_catalog import get_tournament_context

    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    tournaments: List[str] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "winner_name" not in fieldnames or "loser_name" not in fieldnames:
            return [], [], None
        for row in reader:
            winner_name = (row.get("winner_name") or "").strip()
            loser_name = (row.get("loser_name") or "").strip()
            if not winner_name or not loser_name:
                continue
            swap = rng.random() < 0.5
            if swap:
                player_a = PlayerStats(
                    name=loser_name,
                    rank=safe_float(row.get("loser_rank")),
                    points=safe_float(row.get("loser_rank_points")),
                    serve_rating=None,
                    return_rating=None,
                    pressure_rating=None,
                    winloss_index=None,
                    tdi_perf=None,
                )
                player_b = PlayerStats(
                    name=winner_name,
                    rank=safe_float(row.get("winner_rank")),
                    points=safe_float(row.get("winner_rank_points")),
                    serve_rating=None,
                    return_rating=None,
                    pressure_rating=None,
                    winloss_index=None,
                    tdi_perf=None,
                )
                label = 0
            else:
                player_a = PlayerStats(
                    name=winner_name,
                    rank=safe_float(row.get("winner_rank")),
                    points=safe_float(row.get("winner_rank_points")),
                    serve_rating=None,
                    return_rating=None,
                    pressure_rating=None,
                    winloss_index=None,
                    tdi_perf=None,
                )
                player_b = PlayerStats(
                    name=loser_name,
                    rank=safe_float(row.get("loser_rank")),
                    points=safe_float(row.get("loser_rank_points")),
                    serve_rating=None,
                    return_rating=None,
                    pressure_rating=None,
                    winloss_index=None,
                    tdi_perf=None,
                )
                label = 1
            tourney_name = (row.get("tourney_name") or "").strip()
            tournament_ctx = get_tournament_context(tourney_name, catalog) if catalog else None
            if tournament_ctx is None and (row.get("surface") or row.get("tourney_date")):
                surface = _surface_from_atp(
                    row.get("surface") or "",
                    row.get("indoor") or "",
                )
                date_str = (row.get("tourney_date") or "")[:6]
                month = int(date_str[4:6]) if len(date_str) >= 6 else 0
                try:
                    month = max(1, min(12, month))
                except (ValueError, TypeError):
                    month = 0
                tournament_ctx = {
                    "surface": surface,
                    "month": month,
                    "temperature_avg": None,
                    "humidity_avg": None,
                    "surface_speed_rating": None,
                    "court_type": "indoor" if "indoor" in surface else "outdoor",
                    "altitude_m": None,
                }
            if tournament_ctx is None:
                tournament_ctx = {}
            if row.get("best_of") is not None:
                tournament_ctx["best_of"] = safe_float(row.get("best_of"))
            if row.get("draw_size") is not None:
                tournament_ctx["draw_size"] = safe_float(row.get("draw_size"))
            if row.get("round"):
                tournament_ctx["round"] = (row.get("round") or "").strip()
            match_ctx = _match_context_from_row(row, swap=swap)
            if tournament_ctx:
                match_ctx["tournament_country"] = tournament_ctx.get("country")
            fs = feature_set if feature_set in ("core", "enhanced", "legacy") else "core"
            mf = build_match_features(
                player_a, player_b,
                tournament_context=tournament_ctx,
                medians=DEFAULT_MEDIANS,
                match_context=match_ctx if any(v is not None and v != "" for v in match_ctx.values()) else None,
                feature_set=fs,
            )
            md = _parse_row_date(row)
            rows.append({"feature_vector": mf.feature_vector, "match_date": md})
            labels.append(label)
            tournaments.append(tourney_name or "")

    return rows, labels, tournaments


def load_historical_csv(
    path: str,
    catalog: Optional[Any] = None,
    feature_set: str = "legacy",
    require_atp_coverage_for_enhanced: bool = True,
    mirror_rows: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], Optional[List[str]]]:
    """
    Load CSV with columns suitable for training.
    feature_set: "core" | "enhanced" | "legacy". For "enhanced", when require_atp_coverage_for_enhanced
    is True, only rows where both players have at least 2 of (serve_rating, return_rating, pressure_rating) are included.
    """
    import csv
    from deuce.tournament_catalog import get_tournament_context
    from deuce.training.features import has_atp_leaderboard_coverage

    rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    tournaments: List[str] = []
    has_tournament = False

    def _swap_match_ctx(mc: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(mc)
        a_keys = [k for k in out.keys() if k.endswith("_a")]
        for k_a in a_keys:
            k_b = k_a[:-2] + "_b"
            if k_b in out:
                out[k_a], out[k_b] = out[k_b], out[k_a]
        return out

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], [], None
        fieldnames = (reader.fieldnames or [])
        has_tournament = "tournament" in fieldnames
        for row in reader:
            winner = (row.get("winner") or "").strip().upper()
            if winner not in ("A", "B"):
                continue
            label = 1 if winner == "A" else 0

            player_a = PlayerStats(
                name=row.get("player_a", "").strip() or "A",
                rank=safe_float(row.get("rank_a")),
                points=safe_float(row.get("points_a")),
                serve_rating=safe_float(row.get("serve_rating_a")),
                return_rating=safe_float(row.get("return_rating_a")),
                pressure_rating=safe_float(row.get("pressure_rating_a")),
                winloss_index=safe_float(row.get("winloss_index_a")),
                tdi_perf=safe_float(row.get("tdi_perf_a")),
            )
            player_b = PlayerStats(
                name=row.get("player_b", "").strip() or "B",
                rank=safe_float(row.get("rank_b")),
                points=safe_float(row.get("points_b")),
                serve_rating=safe_float(row.get("serve_rating_b")),
                return_rating=safe_float(row.get("return_rating_b")),
                pressure_rating=safe_float(row.get("pressure_rating_b")),
                winloss_index=safe_float(row.get("winloss_index_b")),
                tdi_perf=safe_float(row.get("tdi_perf_b")),
            )
            if feature_set == "enhanced" and require_atp_coverage_for_enhanced:
                if not has_atp_leaderboard_coverage(player_a, player_b, min_required=2):
                    continue
            tournament_name = (row.get("tournament") or "").strip() if has_tournament else ""
            tournament_ctx = get_tournament_context(tournament_name, catalog) if (catalog and tournament_name) else None
            tournament_ctx = dict(tournament_ctx) if tournament_ctx else {}
            if row.get("surface"):
                tournament_ctx["surface"] = (row.get("surface") or "").strip()
            if row.get("indoor") is not None and str(row.get("indoor")).strip() in ("1", "I", "true"):
                tournament_ctx["court_type"] = "indoor"
            if row.get("best_of") is not None:
                tournament_ctx["best_of"] = safe_float(row.get("best_of"))
            if row.get("draw_size") is not None:
                tournament_ctx["draw_size"] = safe_float(row.get("draw_size"))
            if row.get("round"):
                tournament_ctx["round"] = (row.get("round") or "").strip()
            match_ctx = _match_context_from_historical_row(row)
            if tournament_ctx:
                match_ctx["tournament_country"] = tournament_ctx.get("country")
            mf = build_match_features(
                player_a, player_b,
                tournament_context=tournament_ctx or None,
                medians=DEFAULT_MEDIANS,
                match_context=match_ctx if any(match_ctx.values()) else None,
                feature_set=feature_set if feature_set in ("core", "enhanced", "legacy") else "legacy",
            )
            md = _parse_row_date(row)
            rows.append({"feature_vector": mf.feature_vector, "match_date": md})
            labels.append(label)
            if has_tournament:
                tournaments.append(tournament_name)

            # Mirror augmentation to enforce player-order symmetry:
            # add (B vs A) with inverted label and swapped match-context fields.
            if mirror_rows:
                mc_swap = _swap_match_ctx(match_ctx)
                mf_swap = build_match_features(
                    player_b, player_a,
                    tournament_context=tournament_ctx or None,
                    medians=DEFAULT_MEDIANS,
                    match_context=mc_swap if any(mc_swap.values()) else None,
                    feature_set=feature_set if feature_set in ("core", "enhanced", "legacy") else "legacy",
                )
                rows.append({"feature_vector": mf_swap.feature_vector, "match_date": md})
                labels.append(1 - label)
                if has_tournament:
                    tournaments.append(tournament_name)

    return rows, labels, tournaments if has_tournament else None


def train_and_save(
    data_path: str,
    out_path: str,
    *,
    val_frac: float = 0.2,
    random_state: int = 42,
    catalog_path: Optional[str] = None,
    model_type: str = "legacy",
    mirror_rows: Optional[bool] = None,
    split_strategy: str = "random",
    calibration_method: str = "platt",
    test_frac: float = 0.0,
) -> None:
    """
    Load CSV, train logistic regression, save .pkl and .features.json.
    model_type: "core" (pre-match, no score/ATP leakage), "enhanced" (core + ATP when coverage),
    or "legacy" (full 127 features). For "enhanced" only rows with both players having ATP stats are used.

    split_strategy: random | time | walkforward (walkforward uses same ordering as time for single split).
    calibration_method: none | platt | isotonic (fit on validation only).
    test_frac: optional holdout fraction (time-ordered tail); only applied when split_strategy is time/walkforward.
    """
    if np is None or LogisticRegression is None:
        raise RuntimeError(
            "Training requires numpy and scikit-learn. "
            "Install with: pip install numpy scikit-learn"
        ) from getattr(sys.modules[__name__], "_import_err", None)

    catalog = None
    if catalog_path and Path(catalog_path).exists():
        from deuce.tournament_catalog import load_catalog
        catalog = load_catalog(catalog_path)

    feature_set = "legacy"
    feature_names: List[str] = FEATURE_NAMES
    if model_type == "core":
        feature_set = "core"
        feature_names = CORE_FEATURE_NAMES
        assert_no_leakage_in_feature_list(feature_names, context="core training")
    elif model_type == "enhanced":
        feature_set = "enhanced"
        feature_names = ENHANCED_FEATURE_NAMES
        assert_no_leakage_in_feature_list(feature_names, context="enhanced training")

    if mirror_rows is None:
        # Default: enforce symmetry for core/enhanced pre-match models.
        mirror_rows = model_type in ("core", "enhanced")

    rows, labels, tournaments = load_csv_auto(
        data_path, catalog,
        feature_set=feature_set,
        require_atp_coverage_for_enhanced=(model_type == "enhanced"),
        mirror_rows=bool(mirror_rows),
    )
    if not rows or not labels:
        raise ValueError(f"No valid rows in {data_path} (model_type={model_type}). For enhanced, ensure CSV has serve_rating_a/b, return_rating_a/b, etc.).")

    X = np.array([r["feature_vector"] for r in rows], dtype=float)
    y = np.array(labels, dtype=int)
    dates = np.array([int(r.get("match_date") or 0) for r in rows], dtype=np.int64)

    # Optional sample weights by tournament tier (Grand Slam > Masters 1000 > 500 > 250 > Challenger)
    sample_weight: Optional[np.ndarray] = None
    if catalog and tournaments:
        from deuce.tournament_catalog import get_weight_for_match
        if catalog:
            weights = [get_weight_for_match(t, catalog=catalog) for t in tournaments]
            sample_weight = np.array(weights, dtype=float)
    elif tournaments:
        # No catalog file: infer weight from tournament name only
        from deuce.tournament_catalog import get_weight_for_match
        weights = [get_weight_for_match(t, catalog=None) for t in tournaments]
        sample_weight = np.array(weights, dtype=float)

    return _run_training_pipeline(
        X,
        y,
        dates,
        feature_names=feature_names,
        model_type=model_type,
        val_frac=val_frac,
        random_state=random_state,
        split_strategy=split_strategy,
        calibration_method=calibration_method,
        test_frac=test_frac,
        sample_weight=sample_weight,
        out_path=out_path,
        train_files=[str(Path(data_path).resolve())],
        primary_data_ref=str(Path(data_path).resolve()),
    )


def _run_training_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    feature_names: List[str],
    model_type: str,
    val_frac: float,
    random_state: int,
    split_strategy: str,
    calibration_method: str,
    test_frac: float,
    sample_weight: Optional[np.ndarray],
    out_path: str,
    train_files: List[str],
    primary_data_ref: str,
) -> Dict[str, Any]:
    """Shared train/calibrate/save path for single-file and data-dir training."""
    dates_full = dates.copy()
    n = len(y)
    idx = np.arange(n)
    strat = (split_strategy or "random").lower()
    if strat == "walkforward":
        strat = "time"

    i_test = np.array([], dtype=np.int64)
    if test_frac > 0 and strat in ("time",) and split_train_val_indices is not None:
        order = np.argsort(dates, kind="stable")
        n_test = max(1, int(n * test_frac))
        if n_test >= n - 3:
            print("Warning: test_frac too large; skipping test holdout.", file=sys.stderr)
        else:
            i_test = order[-n_test:]
            idx = order[:-n_test]
            n = len(idx)
            X = X[idx]
            y = y[idx]
            dates = dates[idx]
            if sample_weight is not None:
                sample_weight = sample_weight[idx]

    effective_split = strat if strat in ("random", "time") else "random"
    split_used = effective_split
    i_train: np.ndarray
    i_val: np.ndarray
    if split_train_val_indices is None:
        from collections import Counter

        counts = Counter(int(v) for v in y)
        use_split = val_frac > 0 and val_frac < 1 and n >= 4 and min(counts.values()) >= 2
        if use_split:
            i_train, i_val = train_test_split(
                np.arange(n), test_size=val_frac, random_state=random_state, stratify=y
            )
        else:
            i_train = np.arange(n)
            i_val = np.arange(n)
        split_used = "random_sklearn"
    else:
        try:
            i_train, i_val = split_train_val_indices(
                n, y, dates,
                strategy=effective_split,  # type: ignore[arg-type]
                val_frac=val_frac,
                random_state=random_state,
            )
        except Exception as e:
            print(f"Warning: split failed ({e}); using sklearn random split.", file=sys.stderr)
            from collections import Counter

            counts = Counter(int(v) for v in y)
            use_split = val_frac > 0 and val_frac < 1 and n >= 4 and min(counts.values()) >= 2
            if use_split:
                i_train, i_val = train_test_split(
                    np.arange(n), test_size=val_frac, random_state=random_state, stratify=y
                )
            else:
                i_train = np.arange(n)
                i_val = np.arange(n)
            split_used = "random_fallback"

    if effective_split == "time" and split_used != "random_fallback":
        usable_dates = int(np.sum(dates > 0))
        if usable_dates < max(4, int(0.5 * n)):
            print(
                "Warning: few usable match dates; time split may be unreliable. Prefer random or fix dates.",
                file=sys.stderr,
            )

    X_train, X_val = X[i_train], X[i_val]
    y_train, y_val = y[i_train], y[i_val]
    sw_train = sample_weight[i_train] if sample_weight is not None else None

    cal_m = (calibration_method or "platt").lower()
    if cal_m not in ("none", "platt", "isotonic"):
        cal_m = "platt"

    X_empty_test = np.empty((0, X.shape[1]), dtype=float)
    fold_out = fit_calibrated_lr_fold(
        X_train,
        y_train,
        X_val,
        y_val,
        X_empty_test,
        None,
        calibration_method=cal_m,
        random_state=random_state,
        sample_weight_train=sw_train,
    )
    scaler = fold_out["scaler"]
    model = fold_out["model"]
    calibrated = fold_out["calibrated_model"]
    cal_m = str(fold_out["calibration_method"])
    raw_ll = float(fold_out["val_log_loss_raw"])
    ll = float(fold_out["val_log_loss_calibrated"])
    acc = float(fold_out["val_accuracy"])

    miss_train = _missingness_summary(X_train, feature_names)
    train_date_window = _date_window_str(dates, i_train)
    val_date_window = _date_window_str(dates, i_val)
    test_date_window = _date_window_str(dates_full, i_test) if i_test.size > 0 else {"min": None, "max": None}
    feature_set_name = {"core": "core_prematch", "enhanced": "enhanced_prematch", "legacy": "legacy_full"}.get(
        model_type, str(model_type)
    )

    print(f"Split: {split_used} | calibration: {cal_m}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Raw val log loss: {raw_ll:.4f} | Calibrated log loss: {ll:.4f}")
    print(f"Accuracy: {acc:.4f}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    created = datetime.now(timezone.utc).isoformat()
    bundle = {
        "version": 3,
        "model_type": model_type,
        "model_tier": model_type,
        "feature_set_name": feature_set_name,
        "model_class": "sklearn.linear_model.LogisticRegression",
        "scaler": scaler,
        "model": model,
        "calibrated_model": calibrated,
        "feature_names": feature_names,
        "medians": DEFAULT_MEDIANS,
        "created_at": created,
        "code_version": CODE_VERSION,
        "train_files": train_files,
        "train_data_path": primary_data_ref,
        "split_strategy": split_used,
        "calibration_requested": cal_m,
        "calibration": {
            "method": cal_m,
            "val_log_loss_raw": float(raw_ll),
            "val_log_loss_calibrated": float(ll),
            "val_accuracy": float(acc),
        },
        "train_date_window": train_date_window,
        "validation_date_window": val_date_window,
        "test_date_window": test_date_window,
        "n_samples_total": int(X.shape[0]),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test_holdout": int(i_test.size),
        "missingness_summary_train": miss_train,
    }
    with open(out, "wb") as f:
        pickle.dump(bundle, f)

    features_json = out.with_suffix(out.suffix + ".features.json")
    with open(features_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_names": feature_names,
                "model_type": model_type,
                "feature_set_name": feature_set_name,
                "created_at": created,
                "train_data_path": primary_data_ref,
                "train_files": train_files,
                "split_strategy": split_used,
                "calibration": cal_m,
                "train_date_window": train_date_window,
                "validation_date_window": val_date_window,
                "test_date_window": test_date_window,
                "code_version": CODE_VERSION,
            },
            f,
            indent=2,
        )

    print(f"Saved model {model_type} ({len(feature_names)} features, scaled + calibrated) to {out}")
    print(f"Saved feature order to {features_json}")
    return {"n_train": len(y_train), "n_val": len(y_val), "accuracy": float(acc), "log_loss": float(ll)}


def load_csv_auto(
    path: str,
    catalog: Optional[Any],
    feature_set: str = "legacy",
    require_atp_coverage_for_enhanced: bool = True,
    mirror_rows: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], Optional[List[str]]]:
    """Load CSV: ATP format (winner_name, loser_name) or training format (player_a, player_b, winner)."""
    import csv
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    if "winner_name" in fieldnames and "loser_name" in fieldnames:
        return load_atp_match_csv(path, catalog=catalog, feature_set=feature_set)
    return load_historical_csv(
        path, catalog=catalog,
        feature_set=feature_set,
        require_atp_coverage_for_enhanced=require_atp_coverage_for_enhanced,
        mirror_rows=mirror_rows,
    )


def train_from_data_dir(
    data_dir: str,
    out_path: str,
    *,
    val_frac: float = 0.2,
    random_state: int = 42,
    catalog_path: Optional[str] = None,
    model_type: str = "legacy",
    split_strategy: str = "time",
    calibration_method: str = "platt",
    test_frac: float = 0.0,
) -> Dict[str, Any]:
    """
    Iterate over all 20XX.csv and 20XX_challenger.csv (2000-2026) in data_dir, load each with load_csv_auto,
    aggregate rows/labels/tournaments, train, save model, return summary.
    model_type: "core", "enhanced", or "legacy" (same as train_and_save).
    """
    if np is None or LogisticRegression is None:
        raise RuntimeError(
            "Training requires numpy and scikit-learn. "
            "Install with: pip install numpy scikit-learn"
        ) from getattr(sys.modules[__name__], "_import_err", None)

    catalog = None
    if catalog_path and Path(catalog_path).exists():
        from deuce.tournament_catalog import load_catalog
        catalog = load_catalog(catalog_path)

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")

    all_rows: List[Dict[str, Any]] = []
    all_labels: List[int] = []
    all_tournaments: List[str] = []

    files: List[Path] = []
    for y in range(2000, 2027):
        for base in (f"{y}.csv", f"{y}_challenger.csv"):
            p = data_path / base
            if p.exists():
                files.append(p)
    files.sort()

    if not files:
        raise ValueError(f"No 20XX.csv or 20XX_challenger.csv files found in {data_dir}")

    feature_set = "legacy"
    feature_names: List[str] = FEATURE_NAMES
    if model_type == "core":
        feature_set = "core"
        feature_names = CORE_FEATURE_NAMES
        assert_no_leakage_in_feature_list(feature_names, context="core training (data-dir)")
    elif model_type == "enhanced":
        feature_set = "enhanced"
        feature_names = ENHANCED_FEATURE_NAMES
        assert_no_leakage_in_feature_list(feature_names, context="enhanced training (data-dir)")

    mirror = model_type in ("core", "enhanced")
    print(f"Loading {len(files)} files from {data_dir} (model_type={model_type})...", file=sys.stderr)
    for p in files:
        try:
            rows, labels, tournaments = load_csv_auto(
                str(p), catalog,
                feature_set=feature_set,
                require_atp_coverage_for_enhanced=(model_type == "enhanced"),
                mirror_rows=mirror,
            )
            if rows and labels:
                all_rows.extend(rows)
                all_labels.extend(labels)
                all_tournaments.extend(tournaments or [""] * len(rows))
                print(f"  {p.name}: {len(rows)} matches", file=sys.stderr)
        except Exception as e:
            print(f"  {p.name}: skip ({e})", file=sys.stderr)

    if not all_rows or not all_labels:
        raise ValueError("No rows loaded from any file")

    X = np.array([r["feature_vector"] for r in all_rows], dtype=float)
    y = np.array(all_labels, dtype=int)
    dates = np.array([int(r.get("match_date") or 0) for r in all_rows], dtype=np.int64)
    tournaments = all_tournaments

    sample_weight: Optional[np.ndarray] = None
    if catalog and tournaments:
        from deuce.tournament_catalog import get_weight_for_match
        weights = [get_weight_for_match(t or "unknown", catalog=catalog) for t in tournaments]
        sample_weight = np.array(weights, dtype=float)
    elif tournaments:
        from deuce.tournament_catalog import get_weight_for_match
        weights = [get_weight_for_match(t or "unknown", catalog=None) for t in tournaments]
        sample_weight = np.array(weights, dtype=float)

    train_files = [str(p.resolve()) for p in files]
    primary_ref = str(Path(data_dir).resolve())
    print(f"Total samples: {len(y)} from {len(files)} files", file=sys.stderr)

    summary = _run_training_pipeline(
        X,
        y,
        dates,
        feature_names=feature_names,
        model_type=model_type,
        val_frac=val_frac,
        random_state=random_state,
        split_strategy=split_strategy,
        calibration_method=calibration_method,
        test_frac=test_frac,
        sample_weight=sample_weight,
        out_path=out_path,
        train_files=train_files,
        primary_data_ref=primary_ref,
    )
    # Enrich bundle on disk with data-dir metadata
    import pickle

    with open(out_path, "rb") as f:
        b = pickle.load(f)
    if isinstance(b, dict):
        b["train_data_dir"] = primary_ref
        b["n_files_loaded"] = len(files)
    with open(out_path, "wb") as f:
        pickle.dump(b, f)

    return {
        "n_total": X.shape[0],
        "n_train": summary["n_train"],
        "n_val": summary["n_val"],
        "n_files": len(files),
        "accuracy": summary["accuracy"],
        "log_loss": summary["log_loss"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train tennis match outcome model")
    parser.add_argument(
        "--data",
        default=None,
        help="Path to a single historical matches CSV",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to directory containing 2000.csv..2026.csv and 2000_challenger.csv..2026_challenger.csv (iterate all)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/tennis_model.pkl",
        help="Output path for model pickle (default: artifacts/tennis_model.pkl)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction for validation split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Path to tournament_catalog.json for sample weights by tier (optional)",
    )
    parser.add_argument(
        "--model-type",
        choices=("core", "enhanced", "legacy"),
        default="legacy",
        help="Model type: core (pre-match, no score/ATP leakage), enhanced (core + ATP when coverage), legacy (full 127 features). Default: legacy.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable mirrored-row augmentation (A vs B plus B vs A). By default, core/enhanced training mirrors rows to remove player-slot bias.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("random", "time", "walkforward"),
        default="random",
        help="Validation split: random (stratified), time (chronological), walkforward (same ordering as time for single split).",
    )
    parser.add_argument(
        "--calibration",
        choices=("none", "platt", "isotonic"),
        default="platt",
        help="Post-hoc calibration fit on validation predictions only.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.0,
        help="Optional time-ordered test holdout fraction (only with time/walkforward split).",
    )
    args = parser.parse_args()

    if args.data_dir:
        if not Path(args.data_dir).is_dir():
            print(f"Error: data directory not found: {args.data_dir}", file=sys.stderr)
            return 1
        out_path = args.out
        if args.model_type != "legacy" and out_path == "artifacts/tennis_model.pkl":
            out_path = f"artifacts/tennis_model_{args.model_type}.pkl"
        try:
            summary = train_from_data_dir(
                args.data_dir,
                out_path,
                val_frac=args.val_frac,
                random_state=args.random_state,
                catalog_path=args.catalog,
                model_type=args.model_type,
                split_strategy=args.split_strategy,
                calibration_method=args.calibration,
                test_frac=args.test_frac,
            )
            print("\n--- Training summary ---")
            print(f"  Files loaded: {summary['n_files']}")
            print(f"  Total samples: {summary['n_total']}")
            print(f"  Train / Val: {summary['n_train']} / {summary['n_val']}")
            print(f"  Accuracy (validation): {summary['accuracy']:.4f}")
            print(f"  Log loss (validation): {summary['log_loss']:.4f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    if not args.data:
        print("Error: provide either --data <file> or --data-dir <dir>", file=sys.stderr)
        return 1
    if not Path(args.data).exists():
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        return 1
    out_path = args.out
    if args.model_type != "legacy" and out_path == "artifacts/tennis_model.pkl":
        out_path = f"artifacts/tennis_model_{args.model_type}.pkl"
    try:
        train_and_save(
            args.data,
            out_path,
            val_frac=args.val_frac,
            random_state=args.random_state,
            catalog_path=args.catalog,
            model_type=args.model_type,
            mirror_rows=(False if args.no_mirror else None),
            split_strategy=args.split_strategy,
            calibration_method=args.calibration,
            test_frac=args.test_frac,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
