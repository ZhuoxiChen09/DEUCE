"""Walk-forward fold generation and aggregated metrics."""

from __future__ import annotations

import numpy as np

from deuce.research.walk_forward import run_walk_forward_from_arrays
from deuce.training.temporal import (
    assert_calibration_indices_disjoint_from_test,
    assert_fold_train_val_oos_temporal,
    fold_sets_disjoint,
)
from deuce.training.walk_forward_folds import iter_expanding_oos_folds, iter_rolling_oos_folds


def test_expanding_folds_temporal():
    d = np.array([20240101 + i for i in range(60)], dtype=np.int64)
    folds = list(iter_expanding_oos_folds(d, min_pool_rows=20, val_frac=0.2, n_oos_folds=3))
    assert len(folds) == 3
    for f in folds:
        assert_calibration_indices_disjoint_from_test(f.i_val_cal, f.i_oos)
        assert_fold_train_val_oos_temporal(d, f.i_train_fit, f.i_val_cal, f.i_oos)
        assert fold_sets_disjoint(f.i_train_fit, f.i_val_cal, f.i_oos)


def test_rolling_folds():
    d = np.array([20240101 + i for i in range(50)], dtype=np.int64)
    folds = list(iter_rolling_oos_folds(d, train_rows=15, val_rows=5, oos_rows=5, step=5))
    assert len(folds) >= 2


def test_walk_forward_aggregate_metrics():
    rng = np.random.RandomState(0)
    n = 70
    X = rng.randn(n, 20)
    y = (rng.rand(n) > 0.45).astype(int)
    dates = np.array([20240101 + i for i in range(n)], dtype=np.int64)

    rep = run_walk_forward_from_arrays(
        X,
        y,
        dates,
        mode="expanding",
        min_pool_rows=25,
        n_oos_folds=3,
        val_frac=0.2,
        calibration_method="platt",
        random_state=1,
    )
    assert rep.n_folds == 3
    assert rep.n_oos_predictions > 0
    assert np.isfinite(rep.log_loss)
    assert len(rep.folds) == 3
    assert len(rep.oos_rows) == rep.n_oos_predictions
