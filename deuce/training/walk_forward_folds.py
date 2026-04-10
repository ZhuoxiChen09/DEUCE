"""
Walk-forward fold index generation (expanding and rolling windows by date).

Train / calibration / OOS index sets are disjoint; calibration uses only validation rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from deuce.training.temporal import (
    assert_calibration_indices_disjoint_from_test,
    assert_fold_train_val_oos_temporal,
    assert_strictly_ordered_dates,
)

WalkMode = Literal["expanding", "rolling"]


@dataclass
class WalkForwardFoldSpec:
    """One fold: fit LR on train_fit, calibrate on val_cal, evaluate on oos."""

    fold_index: int
    i_train_fit: np.ndarray
    i_val_cal: np.ndarray
    i_oos: np.ndarray


def _sorted_order(dates: np.ndarray) -> np.ndarray:
    return np.argsort(np.asarray(dates, dtype=np.int64), kind="stable")


def iter_expanding_oos_folds(
    dates: np.ndarray,
    *,
    min_pool_rows: int,
    val_frac: float,
    n_oos_folds: int,
    require_positive_dates: bool = True,
) -> Iterator[WalkForwardFoldSpec]:
    """
    Expanding window: pool = all rows strictly before the current OOS block (time-ordered).
    Within pool: last ``val_frac`` of rows (time order) = validation for calibration;
    preceding rows = training fit.

    OOS blocks partition the tail of the time-ordered data (after ``min_pool_rows`` minimum
    rows in the first pool — the first fold uses pool ``sorted_idx[0:s]`` where ``s`` starts
    after min_pool_rows).

    Yields one spec per OOS segment.
    """
    n = len(dates)
    d = np.asarray(dates, dtype=np.int64)
    if require_positive_dates and np.sum(d > 0) < max(4, n // 2):
        raise ValueError("Too many missing dates for walk-forward; need mostly valid tourney_date.")

    order = _sorted_order(d)
    assert_strictly_ordered_dates(d, order, context="walk-forward expanding")

    # Minimum index position `s` in sorted order such that pool [0:s) can split train/val
    min_val = max(1, int(min_pool_rows * max(val_frac, 0.05)))
    if min_pool_rows < min_val + 2:
        min_pool_rows = min_val + 2

    rest = n - min_pool_rows
    if rest < n_oos_folds:
        raise ValueError(
            f"Not enough rows: n={n}, min_pool_rows={min_pool_rows}, need rest >= n_oos_folds ({rest} < {n_oos_folds})"
        )

    fold_sizes = [rest // n_oos_folds] * n_oos_folds
    for i in range(rest % n_oos_folds):
        fold_sizes[i] += 1

    start_oos = min_pool_rows
    for fold_index, fs in enumerate(fold_sizes):
        s = start_oos
        e = s + fs
        i_oos = order[s:e]
        pool = order[0:s]
        if pool.size < min_pool_rows:
            raise ValueError(f"Fold {fold_index}: pool too small ({pool.size})")
        n_val = max(1, int(len(pool) * val_frac))
        if n_val >= len(pool):
            n_val = max(1, len(pool) // 5)
        i_train_fit = pool[:-n_val]
        i_val_cal = pool[-n_val:]
        assert_calibration_indices_disjoint_from_test(i_val_cal, i_oos)
        assert_fold_train_val_oos_temporal(d, i_train_fit, i_val_cal, i_oos)
        yield WalkForwardFoldSpec(fold_index=fold_index, i_train_fit=i_train_fit, i_val_cal=i_val_cal, i_oos=i_oos)
        start_oos = e


def iter_rolling_oos_folds(
    dates: np.ndarray,
    *,
    train_rows: int,
    val_rows: int,
    oos_rows: int,
    step: int,
    require_positive_dates: bool = True,
) -> Iterator[WalkForwardFoldSpec]:
    """
    Rolling window along time-ordered rows: train ``train_rows``, val ``val_rows``, OOS ``oos_rows``,
    advance start by ``step`` each fold.
    """
    n = len(dates)
    d = np.asarray(dates, dtype=np.int64)
    if require_positive_dates and np.sum(d > 0) < max(4, n // 2):
        raise ValueError("Too many missing dates for walk-forward.")

    order = _sorted_order(d)
    assert_strictly_ordered_dates(d, order, context="walk-forward rolling")

    block = train_rows + val_rows + oos_rows
    if block > n or train_rows < 2 or val_rows < 1 or oos_rows < 1:
        raise ValueError("Invalid rolling dimensions.")

    fold_index = 0
    base = 0
    while base + block <= n:
        i_train_fit = order[base : base + train_rows]
        i_val_cal = order[base + train_rows : base + train_rows + val_rows]
        i_oos = order[base + train_rows + val_rows : base + block]
        assert_calibration_indices_disjoint_from_test(i_val_cal, i_oos)
        assert_fold_train_val_oos_temporal(d, i_train_fit, i_val_cal, i_oos)
        yield WalkForwardFoldSpec(fold_index=fold_index, i_train_fit=i_train_fit, i_val_cal=i_val_cal, i_oos=i_oos)
        fold_index += 1
        base += step
