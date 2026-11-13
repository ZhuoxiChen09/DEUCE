"""
Walk-forward retraining: refit + calibrate per fold, aggregate OOS metrics.

Uses ``model_train.fit_calibrated_lr_fold`` (calibration on validation only).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deuce.backtest.engine import BacktestConfig, run_backtest
from deuce.research.report import ResearchReport, research_report_from_rows
from deuce.training.walk_forward_folds import WalkForwardFoldSpec, WalkMode, iter_expanding_oos_folds, iter_rolling_oos_folds


@dataclass
class FoldCalibrationSummary:
    """Per-fold validation calibration diagnostics (no test leakage)."""

    method: str
    val_log_loss_raw: float
    val_log_loss_calibrated: float
    val_accuracy: float


@dataclass
class WalkForwardFoldResult:
    fold_index: int
    n_train_fit: int
    n_val_cal: int
    n_oos: int
    train_date_window: Dict[str, Optional[int]]
    val_date_window: Dict[str, Optional[int]]
    oos_date_window: Dict[str, Optional[int]]
    calibration: FoldCalibrationSummary
    oos_log_loss: float
    oos_brier: float


@dataclass
class BankrollSummary:
    """Subset of backtest PnL metrics for reporting (typed view)."""

    roi: float
    yield_per_bet: float
    max_drawdown: float
    avg_edge: float
    hit_rate: float
    n_bets: int


@dataclass
class WalkForwardAggregateReport:
    """Combined OOS predictions across folds."""

    mode: str
    n_folds: int
    n_oos_predictions: int
    log_loss: float
    brier: float
    roi: float
    yield_per_bet: float
    max_drawdown: float
    avg_edge: float
    hit_rate: float
    bankroll: BankrollSummary
    research: ResearchReport
    folds: List[WalkForwardFoldResult] = field(default_factory=list)
    calibration_mean_val_ll_calibrated: float = 0.0
    oos_rows: List[Dict[str, Any]] = field(default_factory=list)


def _date_win(dates: np.ndarray, idx: np.ndarray) -> Dict[str, Optional[int]]:
    if idx.size == 0:
        return {"min": None, "max": None}
    d = dates[idx]
    d = d[d > 0]
    if d.size == 0:
        return {"min": None, "max": None}
    return {"min": int(np.min(d)), "max": int(np.max(d))}


def run_walk_forward_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    mode: WalkMode = "expanding",
    val_frac: float = 0.2,
    calibration_method: str = "platt",
    random_state: int = 42,
    min_pool_rows: int = 40,
    n_oos_folds: int = 3,
    train_rows: int = 30,
    val_rows: int = 8,
    oos_rows: int = 6,
    rolling_step: int = 4,
    sample_weight: Optional[np.ndarray] = None,
    backtest_cfg: Optional[BacktestConfig] = None,
) -> WalkForwardAggregateReport:
    """
    Run walk-forward retraining on pre-built design matrix and labels (aligned rows).
    """
    import model_train

    d = np.asarray(dates, dtype=np.int64)
    folds: List[WalkForwardFoldSpec] = []
    if mode == "expanding":
        folds = list(
            iter_expanding_oos_folds(
                d,
                min_pool_rows=min_pool_rows,
                val_frac=val_frac,
                n_oos_folds=n_oos_folds,
            )
        )
    else:
        folds = list(
            iter_rolling_oos_folds(
                d,
                train_rows=train_rows,
                val_rows=val_rows,
                oos_rows=oos_rows,
                step=rolling_step,
            )
        )

    oos_rows_out: List[Dict[str, Any]] = []
    fold_results: List[WalkForwardFoldResult] = []
    val_ll_cals: List[float] = []

    for spec in folds:
        tr, va, oos = spec.i_train_fit, spec.i_val_cal, spec.i_oos
        sw_tr = sample_weight[tr] if sample_weight is not None else None
        out = model_train.fit_calibrated_lr_fold(
            X[tr],
            y[tr],
            X[va],
            y[va],
            X[oos],
            y[oos],
            calibration_method=calibration_method,
            random_state=random_state,
            sample_weight_train=sw_tr,
        )
        prob = out["prob_a_test"]
        y_oos = y[oos]
        oos_ll = float(
            -np.mean(
                y_oos * np.log(np.clip(prob, 1e-15, 1.0))
                + (1 - y_oos) * np.log(np.clip(1.0 - prob, 1e-15, 1.0))
            )
        )
        oos_br = float(np.mean((prob - y_oos.astype(float)) ** 2))
        cal = out["calibration"]
        fc = FoldCalibrationSummary(
            method=str(out["calibration_method"]),
            val_log_loss_raw=float(cal["val_log_loss_raw"]),
            val_log_loss_calibrated=float(cal["val_log_loss_calibrated"]),
            val_accuracy=float(cal["val_accuracy"]),
        )
        val_ll_cals.append(fc.val_log_loss_calibrated)
        fold_results.append(
            WalkForwardFoldResult(
                fold_index=spec.fold_index,
                n_train_fit=int(tr.size),
                n_val_cal=int(va.size),
                n_oos=int(oos.size),
                train_date_window=_date_win(d, tr),
                val_date_window=_date_win(d, va),
                oos_date_window=_date_win(d, oos),
                calibration=fc,
                oos_log_loss=oos_ll,
                oos_brier=oos_br,
            )
        )
        for j in range(len(oos)):
            oos_rows_out.append(
                {
                    "model_prob_a": float(prob[j]),
                    "outcome_a_won": int(y_oos[j]),
                    "fold_index": spec.fold_index,
                }
            )

    if not oos_rows_out:
        z = BankrollSummary(0, 0, 0, 0, 0, 0)
        rr = ResearchReport(0, 0, 0, 0, 0, 0, 0, 0, 0)
        return WalkForwardAggregateReport(
            mode=mode,
            n_folds=0,
            n_oos_predictions=0,
            log_loss=0,
            brier=0,
            roi=0,
            yield_per_bet=0,
            max_drawdown=0,
            avg_edge=0,
            hit_rate=0,
            bankroll=z,
            research=rr,
            folds=[],
            calibration_mean_val_ll_calibrated=0.0,
            oos_rows=[],
        )

    rr = research_report_from_rows(oos_rows_out, backtest_cfg=backtest_cfg)
    cfg = backtest_cfg or BacktestConfig(edge_threshold=0.0)
    res_bt = run_backtest(oos_rows_out, config=cfg)
    bank = BankrollSummary(
        roi=res_bt.roi,
        yield_per_bet=res_bt.yield_per_bet,
        max_drawdown=res_bt.max_drawdown,
        avg_edge=res_bt.avg_edge,
        hit_rate=res_bt.hit_rate,
        n_bets=res_bt.n_bets,
    )
    mean_val_ll = float(np.mean(val_ll_cals)) if val_ll_cals else 0.0

    return WalkForwardAggregateReport(
        mode=mode,
        n_folds=len(fold_results),
        n_oos_predictions=len(oos_rows_out),
        log_loss=rr.log_loss,
        brier=rr.brier,
        roi=rr.roi,
        yield_per_bet=rr.yield_per_bet,
        max_drawdown=rr.max_drawdown,
        avg_edge=rr.avg_edge,
        hit_rate=rr.hit_rate,
        bankroll=bank,
        research=rr,
        folds=fold_results,
        calibration_mean_val_ll_calibrated=mean_val_ll,
        oos_rows=oos_rows_out,
    )


def run_walk_forward_from_csv(
    csv_path: str,
    *,
    model_type: str = "core",
    catalog_path: Optional[str] = None,
    mirror_rows: bool = False,
    **kwargs: Any,
) -> WalkForwardAggregateReport:
    """Load CSV via ``model_train.load_csv_auto`` and run walk-forward."""
    import model_train

    catalog = None
    if catalog_path and Path(catalog_path).exists():
        from deuce.tournament_catalog import load_catalog

        catalog = load_catalog(catalog_path)

    fs = "legacy"
    if model_type == "core":
        fs = "core"
    elif model_type == "enhanced":
        fs = "enhanced"

    rows, labels, _ = model_train.load_csv_auto(
        csv_path,
        catalog,
        feature_set=fs,
        require_atp_coverage_for_enhanced=(model_type == "enhanced"),
        mirror_rows=mirror_rows,
    )
    if not rows:
        raise ValueError("No rows loaded from CSV")

    X = np.array([r["feature_vector"] for r in rows], dtype=float)
    y = np.array(labels, dtype=int)
    dates = np.array([int(r.get("match_date") or 0) for r in rows], dtype=np.int64)

    return run_walk_forward_from_arrays(X, y, dates, **kwargs)


def aggregate_report_to_json(rep: WalkForwardAggregateReport) -> Dict[str, Any]:
    """JSON-serializable summary (folds + headline metrics)."""
    cal_summary = {
        "mean_val_log_loss_calibrated": rep.calibration_mean_val_ll_calibrated,
        "per_fold_val_ll_calibrated": [float(f.calibration.val_log_loss_calibrated) for f in rep.folds],
    }
    return {
        "mode": rep.mode,
        "n_folds": rep.n_folds,
        "n_oos_predictions": rep.n_oos_predictions,
        "log_loss": rep.log_loss,
        "brier": rep.brier,
        "roi": rep.roi,
        "yield_per_bet": rep.yield_per_bet,
        "max_drawdown": rep.max_drawdown,
        "avg_edge": rep.avg_edge,
        "hit_rate": rep.hit_rate,
        "calibration_mean_val_ll_calibrated": rep.calibration_mean_val_ll_calibrated,
        "calibration_summary": cal_summary,
        "bankroll": asdict(rep.bankroll),
        "research": {
            "n": rep.research.n,
            "log_loss": rep.research.log_loss,
            "brier": rep.research.brier,
            "by_tour": rep.research.by_tour,
            "by_surface": rep.research.by_surface,
        },
        "folds": [asdict(f) for f in rep.folds],
    }


def save_walk_forward_outputs(
    rep: WalkForwardAggregateReport,
    *,
    out_json: Optional[str] = None,
    out_oos_csv: Optional[str] = None,
) -> None:
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(aggregate_report_to_json(rep), f, indent=2, default=str)
    if out_oos_csv and rep.oos_rows:
        Path(out_oos_csv).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({k for r in rep.oos_rows for k in r.keys()})
        if not fieldnames:
            return
        with open(out_oos_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rep.oos_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
