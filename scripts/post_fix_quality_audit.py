"""
Post-fix model quality audit for the tennis Core model.

Runs:
- Retrain corrected Core with mirrored rows (default) and report metrics
- Coefficient table
- Favorite sanity tests using player_db ranks/points
- Calibration table (probability bins)
- Ablations A–E
- Dead/junk feature checks (zero coef, near-zero variance, high missing)

Run from project root:
  python scripts/post_fix_quality_audit.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class TrainResult:
    feature_names: List[str]
    n_total: int
    n_train: int
    n_val: int
    mirror_rows: bool
    acc_train: float
    acc_val: float
    logloss_train: float
    logloss_val: float
    brier_val: float
    auc_val: Optional[float]
    confusion_val: List[List[int]]
    coef: List[float]
    intercept: float
    missing_rate_by_feature: Dict[str, float]
    var_by_feature: Dict[str, float]
    calib_bins: List[Dict[str, Any]]


def _fit_platt(model: Any, X_val: Any, y_val: Any, *, random_state: int = 42) -> Any:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    raw_prob = model.predict_proba(X_val)[:, 1]
    raw_prob = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
    platt = LogisticRegression(max_iter=500, random_state=random_state)
    platt.fit(raw_prob.reshape(-1, 1), y_val)
    return platt


class _PlattWrapper:
    def __init__(self, base_model: Any, platt: Any):
        self.base_model = base_model
        self.platt = platt
        self.classes_ = getattr(platt, "classes_", [0, 1])

    def predict_proba(self, X: Any) -> Any:
        import numpy as np

        raw = self.base_model.predict_proba(X)[:, 1]
        raw = np.clip(raw, 1e-6, 1.0 - 1e-6)
        return self.platt.predict_proba(raw.reshape(-1, 1))


def _train_eval(
    *,
    X,
    y,
    feature_names: List[str],
    mirror_rows: bool,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        brier_score_loss,
        roc_auc_score,
        confusion_matrix,
    )

    n_samples = int(len(y))
    idx = np.arange(n_samples)
    i_train, i_val = train_test_split(idx, test_size=val_frac, random_state=random_state, stratify=y)
    X_train, X_val = X[i_train], X[i_val]
    y_train, y_val = y[i_train], y[i_val]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_s, y_train)
    platt = _fit_platt(model, X_val_s, y_val, random_state=random_state)
    calibrated = _PlattWrapper(model, platt)

    p_train = calibrated.predict_proba(X_train_s)[:, 1]
    p_val = calibrated.predict_proba(X_val_s)[:, 1]

    pred_train = (p_train >= 0.5).astype(int)
    pred_val = (p_val >= 0.5).astype(int)

    acc_train = float(accuracy_score(y_train, pred_train))
    acc_val = float(accuracy_score(y_val, pred_val))
    ll_train = float(log_loss(y_train, np.clip(p_train, 1e-15, 1 - 1e-15)))
    ll_val = float(log_loss(y_val, np.clip(p_val, 1e-15, 1 - 1e-15)))
    brier_val = float(brier_score_loss(y_val, p_val))
    try:
        auc_val = float(roc_auc_score(y_val, p_val))
    except Exception:
        auc_val = None
    cm = confusion_matrix(y_val, pred_val).tolist()

    # missing rate: if feature has *_missing counterpart, use mean(missing_flag)
    name_to_idx = {fname: i for i, fname in enumerate(feature_names)}
    missing_rate: Dict[str, float] = {}
    for fname in feature_names:
        if fname.endswith("_missing"):
            base = fname[:-8]
            v = X[:, name_to_idx[fname]]
            missing_rate[base] = float(np.mean(v))

    # variance in raw X (before scaling)
    var_by: Dict[str, float] = {feature_names[i]: float(np.var(X[:, i])) for i in range(len(feature_names))}

    # calibration bins
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0000001]
    calib_rows: List[Dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_val >= lo) & (p_val < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        win_rate = float(np.mean(y_val[mask]))
        avg_p = float(np.mean(p_val[mask]))
        calib_rows.append(
            {
                "bucket": f"[{lo:.1f},{hi:.1f})",
                "count": cnt,
                "avg_pred": round(avg_p, 4),
                "actual_win_rate": round(win_rate, 4),
            }
        )

    return TrainResult(
        feature_names=feature_names,
        n_total=n_samples,
        n_train=int(len(y_train)),
        n_val=int(len(y_val)),
        mirror_rows=mirror_rows,
        acc_train=acc_train,
        acc_val=acc_val,
        logloss_train=ll_train,
        logloss_val=ll_val,
        brier_val=brier_val,
        auc_val=auc_val,
        confusion_val=cm,
        coef=[float(x) for x in model.coef_[0]],
        intercept=float(model.intercept_[0]),
        missing_rate_by_feature=missing_rate,
        var_by_feature=var_by,
        calib_bins=calib_rows,
    )


def _select_features(X, all_names: List[str], keep: Sequence[str]) -> Tuple[Any, List[str]]:
    import numpy as np

    idx = [all_names.index(n) for n in keep]
    X2 = X[:, idx]
    return X2, list(keep)


def _core_ablation_sets(core_names: List[str]) -> Dict[str, List[str]]:
    # Convenience groups
    seeds = [
        "seed_diff", "seed_diff_missing",
        "seeded_flag_a", "seeded_flag_a_missing",
        "seeded_flag_b", "seeded_flag_b_missing",
        "both_seeded_flag", "unseeded_vs_seeded_flag",
    ]
    artifacts = ["event_id_hash", "event_id_missing", "match_num", "match_num_missing", "in_event_fatigue_proxy"]
    rank_only = ["rank_diff", "rank_diff_missing"]
    rank_points = ["rank_diff", "rank_diff_missing", "log_points_diff", "log_points_diff_missing"]

    out: Dict[str, List[str]] = {}
    out["A_rank_diff_only"] = rank_only
    out["B_rank_plus_log_points"] = rank_points
    out["C_rank_points_plus_seed"] = [n for n in (rank_points + seeds) if n in core_names]
    out["D_core_no_event_id_match_progression"] = [n for n in core_names if n not in artifacts]
    out["E_full_corrected_core"] = list(core_names)
    return out


def _favorite_sanity_tests() -> List[Tuple[str, str]]:
    # 10 obvious mismatches (A strong, B weaker). These names must exist in the DB to get rank/points.
    return [
        ("Jannik Sinner", "Billy Harris"),
        ("Carlos Alcaraz", "Patrick Kypson"),
        ("Novak Djokovic", "Valentin Vacherot"),
        ("Daniil Medvedev", "Rei Sakamoto"),
        ("Alexander Zverev", "Coleman Wong"),
        ("Andrey Rublev", "Jaime Faria"),
        ("Stefanos Tsitsipas", "Aleksandar Vukic"),
        ("Casper Ruud", "Eliot Spizzirri"),
        ("Taylor Fritz", "Rinky Hijikata"),
        ("Hubert Hurkacz", "Francisco Comesana"),
    ]


def main() -> int:
    import numpy as np

    from model_features import CORE_FEATURE_NAMES, build_player_stats_from_datasets
    from model_predict import predict_match_probability, load_model
    from player_db import DEFAULT_DB_PATH, load_db
    from model_train import load_historical_csv

    data_path = ROOT / "data" / "historical_2026_v3.csv"
    if not data_path.exists():
        print(json.dumps({"error": f"Missing data file: {data_path}"}))
        return 1

    # Load corrected core training data with mirroring enabled.
    rows, labels, _ = load_historical_csv(
        str(data_path),
        catalog=None,
        feature_set="core",
        require_atp_coverage_for_enhanced=False,
        mirror_rows=True,
    )
    X = np.array([r["feature_vector"] for r in rows], dtype=float)
    y = np.array(labels, dtype=int)

    # Train full corrected core
    full = _train_eval(X=X, y=y, feature_names=list(CORE_FEATURE_NAMES), mirror_rows=True)

    # Load the post-fix trained artifact for inference sanity tests
    model_path = ROOT / "artifacts" / "tennis_model_core.pkl"
    bundle, _, _ = load_model(str(model_path)) if model_path.exists() else (None, None, None)

    # Ablations
    ablation_sets = _core_ablation_sets(list(CORE_FEATURE_NAMES))
    ablations: Dict[str, Dict[str, Any]] = {}
    for key, keep in ablation_sets.items():
        Xk, names_k = _select_features(X, list(CORE_FEATURE_NAMES), keep)
        res_k = _train_eval(X=Xk, y=y, feature_names=names_k, mirror_rows=True)
        ablations[key] = {
            "n_total": res_k.n_total,
            "n_train": res_k.n_train,
            "n_val": res_k.n_val,
            "acc_train": res_k.acc_train,
            "acc_val": res_k.acc_val,
            "logloss_val": res_k.logloss_val,
            "brier_val": res_k.brier_val,
            "auc_val": res_k.auc_val,
        }

    # Coef table (full)
    coef_rows = []
    for i, name in enumerate(full.feature_names):
        coef_rows.append({"feature_name": name, "coefficient": full.coef[i], "abs": abs(full.coef[i])})
    coef_rows.sort(key=lambda r: -r["abs"])
    for r, row in enumerate(coef_rows, start=1):
        row["rank"] = r
        row.pop("abs", None)

    # Dead/junk feature checks
    zero_coef = [row["feature_name"] for row in coef_rows if abs(row["coefficient"]) < 1e-9]
    near_zero_var = sorted(
        [{"feature_name": k, "variance": v} for k, v in full.var_by_feature.items()],
        key=lambda x: x["variance"],
    )[:15]
    high_missing = sorted(
        [{"feature_name": k, "missing_rate": v} for k, v in full.missing_rate_by_feature.items()],
        key=lambda x: -x["missing_rate"],
    )[:15]

    # Favorite sanity tests: DB-based pre-match, empty tournament context.
    db = load_db(DEFAULT_DB_PATH)
    sanity = []
    for a, b in _favorite_sanity_tests():
        sa = build_player_stats_from_datasets(a, player_db=db)
        sb = build_player_stats_from_datasets(b, player_db=db)
        res = predict_match_probability(sa, sb, model_core=bundle, model_enhanced=None, model=None)
        # Use the retrained full model by doing an in-memory predict: reuse our trained params is non-trivial,
        # so sanity tests are reported against the app's default loaded model artifact.
        # (The post-fix core artifact should match this training configuration when you run model_train.)
        sanity.append(
            {
                "player_a": a,
                "player_b": b,
                "rank_a": sa.rank,
                "rank_b": sb.rank,
                "points_a": sa.points,
                "points_b": sb.points,
                "p_a_wins": res.model_prob_a,
                "p_b_wins": res.model_prob_b,
                "favored_stronger_by_rank_points": None
                if (sa.rank is None or sb.rank is None)
                else ((a if sa.rank < sb.rank else b) == (a if res.model_prob_a >= 0.5 else b)),
                "confidence": res.confidence,
                "model_used": res.model_used,
                "warnings": res.warnings,
            }
        )

    # Add one symmetry check by swapping the first matchup
    swap_check = None
    if sanity:
        a, b = sanity[0]["player_a"], sanity[0]["player_b"]
        sa = build_player_stats_from_datasets(a, player_db=db)
        sb = build_player_stats_from_datasets(b, player_db=db)
        r1 = predict_match_probability(sa, sb, model_core=bundle, model_enhanced=None, model=None)
        r2 = predict_match_probability(sb, sa, model_core=bundle, model_enhanced=None, model=None)
        swap_check = {
            "matchup": f"{a} vs {b}",
            "p_a_wins_original": r1.model_prob_a,
            "p_a_wins_after_swap": r2.model_prob_a,
            "symmetry_error_abs": abs(r1.model_prob_a - (1.0 - r2.model_prob_a)),
        }

    out = {
        "post_fix_training_results": {
            "n_matches": full.n_total,
            "mirror_rows_used": full.mirror_rows,
            "n_train": full.n_train,
            "n_val": full.n_val,
            "train_accuracy": full.acc_train,
            "val_accuracy": full.acc_val,
            "train_log_loss": full.logloss_train,
            "val_log_loss": full.logloss_val,
            "val_brier": full.brier_val,
            "val_roc_auc": full.auc_val,
            "confusion_matrix_val": full.confusion_val,
        },
        "top_feature_effects": {
            "intercept": full.intercept,
            "coefficients_sorted": coef_rows[:40],
        },
        "calibration_bins_val": full.calib_bins,
        "ablations": ablations,
        "dead_or_junk_features": {
            "zero_coefficient_features": zero_coef,
            "lowest_variance_features": near_zero_var,
            "highest_missing_features": high_missing,
        },
        "favorite_sanity_tests": {
            "matchups": sanity,
            "swap_symmetry_check": swap_check,
            "notes": "Sanity tests use the default artifact loaded by predict_match_probability; ensure artifacts/tennis_model_core.pkl is the post-fix retrain output.",
        },
        "data_file": str(data_path),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

