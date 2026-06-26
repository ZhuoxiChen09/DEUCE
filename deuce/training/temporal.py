"""
Temporal safety checks for time-ordered splits and walk-forward folds.

Use these in tests and research pipelines to catch leakage / ordering bugs.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def _range_for_indices(dates: np.ndarray, idx: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    if idx.size == 0:
        return None, None
    d = np.asarray(dates, dtype=np.int64)[idx]
    d = d[d > 0]
    if d.size == 0:
        return None, None
    return int(np.min(d)), int(np.max(d))


def assert_strictly_ordered_dates(dates: np.ndarray, order: np.ndarray, *, context: str = "") -> None:
    """Raise if dates along ``order`` are not non-decreasing (required for time splits)."""
    d = np.asarray(dates, dtype=np.int64)[order]
    d = d[d > 0]
    if d.size <= 1:
        return
    if not np.all(d[:-1] <= d[1:]):
        msg = "Dates are not sorted along the given order" + (f" ({context})" if context else "")
        raise ValueError(msg)


def assert_disjoint_train_val_test_dates(
    dates: np.ndarray,
    i_train: np.ndarray,
    i_val: np.ndarray,
    i_test: np.ndarray,
    *,
    strict_train_before_val: bool = True,
    strict_val_before_test: bool = True,
) -> None:
    """
    For time splits: train dates should not exceed val min; val should not exceed test min
    (when indices are time-ordered subsets).
    """
    t_lo, t_hi = _range_for_indices(dates, i_train)
    v_lo, v_hi = _range_for_indices(dates, i_val)
    s_lo, s_hi = _range_for_indices(dates, i_test)

    if strict_train_before_val and t_hi is not None and v_lo is not None and t_hi > v_lo:
        raise ValueError(f"Train date max {t_hi} > val date min {v_lo}")
    if strict_val_before_test and v_hi is not None and s_lo is not None and v_hi > s_lo:
        raise ValueError(f"Val date max {v_hi} > test date min {s_lo}")


def assert_fold_train_val_oos_temporal(
    dates: np.ndarray,
    i_train_fit: np.ndarray,
    i_val_cal: np.ndarray,
    i_oos: np.ndarray,
) -> None:
    """
    Expanding walk-forward: all train dates <= all val dates < all OOS dates (strict on positive dates).
    """
    tr_lo, tr_hi = _range_for_indices(dates, i_train_fit)
    va_lo, va_hi = _range_for_indices(dates, i_val_cal)
    o_lo, o_hi = _range_for_indices(dates, i_oos)

    if tr_hi is not None and va_lo is not None and tr_hi > va_lo:
        raise ValueError("Train-fit window overlaps validation (calibration) window in time.")
    if va_hi is not None and o_lo is not None and va_hi >= o_lo:
        raise ValueError("Calibration window must end before the first OOS date.")

    positive_dates = np.asarray(dates, dtype=np.int64)
    train_dates = set(positive_dates[i_train_fit][positive_dates[i_train_fit] > 0].tolist())
    val_dates = set(positive_dates[i_val_cal][positive_dates[i_val_cal] > 0].tolist())
    oos_dates = set(positive_dates[i_oos][positive_dates[i_oos] > 0].tolist())
    if train_dates & val_dates:
        raise ValueError("Train-fit and calibration rows share a calendar date.")
    if train_dates & oos_dates or val_dates & oos_dates:
        raise ValueError("Training/calibration and OOS rows share a calendar date.")


def assert_calibration_indices_disjoint_from_test(i_val: np.ndarray, i_test: np.ndarray) -> None:
    """Calibration must not use test rows."""
    if np.intersect1d(i_val, i_test).size > 0:
        raise ValueError("Calibration indices overlap test indices.")


def assert_no_future_train_for_oos(
    dates: np.ndarray,
    i_train: np.ndarray,
    i_oos: np.ndarray,
) -> None:
    """Every OOS date must be >= train date max (when dates are present)."""
    tr_lo, tr_hi = _range_for_indices(dates, i_train)
    o_lo, o_hi = _range_for_indices(dates, i_oos)
    if tr_hi is not None and o_lo is not None and o_lo < tr_hi:
        raise ValueError("OOS contains dates before end of training window (future leakage).")


def fold_sets_disjoint(
    i_train: np.ndarray,
    i_val: np.ndarray,
    i_test: np.ndarray,
) -> bool:
    """Return True if the three index sets are pairwise disjoint."""
    a = np.intersect1d(i_train, i_val) if i_train.size and i_val.size else np.array([])
    b = np.intersect1d(i_train, i_test) if i_train.size and i_test.size else np.array([])
    c = np.intersect1d(i_val, i_test) if i_val.size and i_test.size else np.array([])
    return a.size == 0 and b.size == 0 and c.size == 0
