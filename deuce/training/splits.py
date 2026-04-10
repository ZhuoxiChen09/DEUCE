"""
Time-aware and random train/validation splits.

Validation indices are always strictly after training indices in time-split mode
(no future data in training set for a given validation point).
"""

from __future__ import annotations

from typing import Any, Iterator, List, Literal, Optional, Tuple

import numpy as np

SplitStrategy = Literal["random", "time", "walkforward"]


def parse_match_date_int(row: Any, keys: Tuple[str, ...] = ("tourney_date", "date", "match_date")) -> int:
    """
    Parse YYYYMMDD or ISO-ish date from a CSV row dict. Returns 0 if missing/invalid.
    """
    if not isinstance(row, dict):
        return 0
    for key in keys:
        v = row.get(key)
        if v is None or v == "":
            continue
        s = str(v).strip().replace("-", "")[:8]
        if len(s) >= 8 and s.isdigit():
            return int(s[:8])
    return 0


def split_train_val_indices(
    n: int,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    strategy: SplitStrategy,
    val_frac: float,
    random_state: int,
    walkforward_val_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (i_train, i_val) index arrays into [0..n).

    - random: stratified shuffle split when possible.
    - time: sort by ``dates`` ascending; first (1-val_frac) for train, last val_frac for val.
    - walkforward: last walkforward_val_frac of *time-ordered* rows as validation (same as time for single split;
      use ``walk_forward_fold_indices`` for multiple folds).
    """
    rng = np.random.RandomState(random_state)
    idx = np.arange(n)

    if strategy == "random":
        from collections import Counter

        counts = Counter(int(v) for v in y)
        use_strat = val_frac > 0 and val_frac < 1 and n >= 4 and min(counts.values()) >= 2
        if use_strat:
            from sklearn.model_selection import train_test_split

            i_train, i_val = train_test_split(
                idx, test_size=val_frac, random_state=random_state, stratify=y
            )
            return np.sort(i_train), np.sort(i_val)
        cut = max(1, int(n * (1.0 - val_frac)))
        return idx[:cut], idx[cut:]

    # time or walkforward: require usable dates
    if dates.size != n:
        raise ValueError("dates array must match n")
    usable = int(np.sum(dates > 0))
    if usable < max(4, int(0.5 * n)):
        # Too many missing dates — fall back to random with warning (caller should log)
        return split_train_val_indices(n, y, dates, strategy="random", val_frac=val_frac, random_state=random_state)

    order = np.argsort(dates, kind="stable")
    sorted_idx = order
    cut = int(n * (1.0 - val_frac))
    cut = max(1, min(cut, n - 1))
    i_train = sorted_idx[:cut]
    i_val = sorted_idx[cut:]
    return i_train, i_val


def walk_forward_fold_indices(
    n: int,
    dates: np.ndarray,
    *,
    n_folds: int,
    min_train_frac: float = 0.5,
) -> Iterator[Tuple[np.ndarray, np.ndarray, Tuple[int, int]]]:
    """
    Yield (i_train, i_val, (val_date_min, val_date_max)) for each fold.

    Expanding window: train on all rows strictly before validation window start.
    Validation windows are sequential chunks of time-ordered data.
    """
    if dates.size != n:
        raise ValueError("dates must match n")
    order = np.argsort(dates, kind="stable")
    sorted_dates = dates[order]
    sorted_idx = order

    min_train = int(n * min_train_frac)
    rest = n - min_train
    if rest < n_folds or n_folds < 2:
        raise ValueError("Not enough rows for walk-forward folds")

    fold_sizes = [rest // n_folds] * n_folds
    for i in range(rest % n_folds):
        fold_sizes[i] += 1

    start = min_train
    pos = 0
    for fs in fold_sizes:
        i_val = sorted_idx[start : start + fs]
        i_train = sorted_idx[:start]
        d_val = sorted_dates[start : start + fs]
        vmin = int(np.min(d_val)) if len(d_val) else 0
        vmax = int(np.max(d_val)) if len(d_val) else 0
        yield i_train, i_val, (vmin, vmax)
        start += fs
        pos += 1
