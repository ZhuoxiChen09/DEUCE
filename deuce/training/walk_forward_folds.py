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


def _grouped_date_indices(dates: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Return stable row-index groups for each unique date, ordered by date."""
    d = np.asarray(dates, dtype=np.int64)
    order = _sorted_order(d)
    sorted_dates = d[order]
    if order.size == 0:
        return [], np.array([], dtype=np.int64)
    boundaries = np.flatnonzero(sorted_dates[1:] != sorted_dates[:-1]) + 1
    groups = [part for part in np.split(order, boundaries) if part.size]
    group_dates = np.array([int(d[group[0]]) for group in groups], dtype=np.int64)
    return groups, group_dates


def _concat_groups(groups: list[np.ndarray], start: int, end: int) -> np.ndarray:
    if start >= end:
        return np.array([], dtype=np.int64)
    return np.concatenate(groups[start:end]).astype(np.int64, copy=False)


def _end_group_for_rows(
    groups: list[np.ndarray],
    start_group: int,
    target_rows: int,
    *,
    max_end_group: int | None = None,
) -> int:
    """Choose a whole-date boundary that reaches approximately ``target_rows``."""
    limit = len(groups) if max_end_group is None else min(max_end_group, len(groups))
    if start_group >= limit:
        return start_group
    target_rows = max(1, int(target_rows))
    rows = 0
    end = start_group
    while end < limit and rows < target_rows:
        rows += len(groups[end])
        end += 1
    return end


def _partition_group_range_by_rows(
    groups: list[np.ndarray],
    start_group: int,
    end_group: int,
    n_parts: int,
) -> list[tuple[int, int]]:
    """Partition contiguous date groups into row-balanced, non-empty parts."""
    n_groups = end_group - start_group
    if n_parts < 1 or n_groups < n_parts:
        raise ValueError(
            f"Need at least one unique date per fold: dates={n_groups}, folds={n_parts}"
        )
    parts: list[tuple[int, int]] = []
    cursor = start_group
    remaining_rows = sum(len(group) for group in groups[start_group:end_group])
    for part_index in range(n_parts):
        remaining_parts = n_parts - part_index
        if remaining_parts == 1:
            next_cursor = end_group
        else:
            max_end = end_group - (remaining_parts - 1)
            target = max(1, round(remaining_rows / remaining_parts))
            next_cursor = _end_group_for_rows(
                groups,
                cursor,
                target,
                max_end_group=max_end,
            )
            next_cursor = max(cursor + 1, min(next_cursor, max_end))
        parts.append((cursor, next_cursor))
        used = sum(len(group) for group in groups[cursor:next_cursor])
        remaining_rows -= used
        cursor = next_cursor
    return parts


def _train_calibration_groups(
    groups: list[np.ndarray],
    pool_end_group: int,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an expanding pool on a whole-date boundary."""
    if pool_end_group < 2:
        raise ValueError("Need at least two unique dates for train/calibration.")
    pool_rows = sum(len(group) for group in groups[:pool_end_group])
    target_val_rows = max(1, int(pool_rows * val_frac))
    val_start = pool_end_group
    val_rows = 0
    while val_start > 1 and val_rows < target_val_rows:
        val_start -= 1
        val_rows += len(groups[val_start])
    return _concat_groups(groups, 0, val_start), _concat_groups(
        groups, val_start, pool_end_group
    )


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
    groups, _ = _grouped_date_indices(d)

    min_val = max(1, int(min_pool_rows * max(val_frac, 0.05)))
    if min_pool_rows < min_val + 2:
        min_pool_rows = min_val + 2
    pool_end_group = _end_group_for_rows(groups, 0, min_pool_rows)
    if pool_end_group < 2:
        raise ValueError("Initial pool must contain at least two unique dates.")
    remaining_groups = len(groups) - pool_end_group
    if remaining_groups < n_oos_folds:
        raise ValueError(
            "Not enough unique OOS dates after the initial pool: "
            f"dates={remaining_groups}, folds={n_oos_folds}"
        )

    oos_group_ranges = _partition_group_range_by_rows(
        groups, pool_end_group, len(groups), n_oos_folds
    )
    for fold_index, (oos_start_group, oos_end_group) in enumerate(oos_group_ranges):
        i_oos = _concat_groups(groups, oos_start_group, oos_end_group)
        pool = _concat_groups(groups, 0, oos_start_group)
        if pool.size < min_pool_rows:
            raise ValueError(f"Fold {fold_index}: pool too small ({pool.size})")
        i_train_fit, i_val_cal = _train_calibration_groups(
            groups, oos_start_group, val_frac
        )
        assert_calibration_indices_disjoint_from_test(i_val_cal, i_oos)
        assert_fold_train_val_oos_temporal(d, i_train_fit, i_val_cal, i_oos)
        yield WalkForwardFoldSpec(fold_index=fold_index, i_train_fit=i_train_fit, i_val_cal=i_val_cal, i_oos=i_oos)


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
    groups, _ = _grouped_date_indices(d)
    if train_rows < 2 or val_rows < 1 or oos_rows < 1:
        raise ValueError("Invalid rolling dimensions.")

    fold_index = 0
    base_group = 0
    while base_group < len(groups):
        train_end = _end_group_for_rows(groups, base_group, train_rows)
        val_end = _end_group_for_rows(groups, train_end, val_rows)
        oos_end = _end_group_for_rows(groups, val_end, oos_rows)
        if train_end <= base_group or val_end <= train_end or oos_end <= val_end:
            break
        i_train_fit = _concat_groups(groups, base_group, train_end)
        i_val_cal = _concat_groups(groups, train_end, val_end)
        i_oos = _concat_groups(groups, val_end, oos_end)
        assert_calibration_indices_disjoint_from_test(i_val_cal, i_oos)
        assert_fold_train_val_oos_temporal(d, i_train_fit, i_val_cal, i_oos)
        yield WalkForwardFoldSpec(fold_index=fold_index, i_train_fit=i_train_fit, i_val_cal=i_val_cal, i_oos=i_oos)
        fold_index += 1
        next_base = _end_group_for_rows(groups, base_group, step)
        if next_base <= base_group:
            break
        base_group = next_base
