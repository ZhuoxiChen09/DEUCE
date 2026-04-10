import numpy as np
import pytest

from deuce.training.splits import parse_match_date_int, split_train_val_indices


def test_parse_date():
    assert parse_match_date_int({"tourney_date": "20240315"}) == 20240315
    assert parse_match_date_int({"date": "2024-03-15"}) == 20240315
    assert parse_match_date_int({}) == 0


def test_time_split_ordering():
    n = 100
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=n)
    dates = np.sort(rng.randint(20100101, 20201231, size=n))
    i_tr, i_val = split_train_val_indices(n, y, dates, strategy="time", val_frac=0.2, random_state=42)
    assert np.max(dates[i_tr]) <= np.min(dates[i_val]) + 1e-9


def test_random_split_stratify():
    n = 50
    y = np.array([0] * 25 + [1] * 25)
    dates = np.zeros(n, dtype=np.int64)
    i_tr, i_val = split_train_val_indices(n, y, dates, strategy="random", val_frac=0.2, random_state=42)
    assert len(i_tr) + len(i_val) == n
