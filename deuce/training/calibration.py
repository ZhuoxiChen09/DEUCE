"""
Post-hoc calibration fit **only on validation** probabilities (never on test).

Methods: none (identity), platt (sigmoid on logit(p)), isotonic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Tuple

import numpy as np

CalibrationMethod = Literal["none", "platt", "isotonic"]


@dataclass
class CalibratorBundle:
    method: CalibrationMethod
    platt: Any = None  # sklearn LogisticRegression on [[logit(p)]]
    isotonic: Any = None  # IsotonicRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def fit_calibration(
    method: CalibrationMethod,
    raw_prob_val: np.ndarray,
    y_val: np.ndarray,
    *,
    random_state: int = 42,
) -> CalibratorBundle:
    if method == "none" or len(raw_prob_val) < 2:
        return CalibratorBundle(method="none")

    raw_prob_val = np.asarray(raw_prob_val, dtype=float).ravel()
    y_val = np.asarray(y_val, dtype=int).ravel()

    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        raw = np.clip(raw_prob_val, 1e-6, 1.0 - 1e-6)
        platt = LogisticRegression(max_iter=500, random_state=random_state)
        platt.fit(raw.reshape(-1, 1), y_val)
        return CalibratorBundle(method="platt", platt=platt)

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_prob_val, y_val)
        return CalibratorBundle(method="isotonic", isotonic=iso)

    return CalibratorBundle(method="none")


def apply_calibration(bundle: CalibratorBundle, raw_prob: np.ndarray) -> np.ndarray:
    raw_prob = np.asarray(raw_prob, dtype=float)
    if bundle.method == "none" or bundle.platt is None and bundle.isotonic is None:
        return np.clip(raw_prob, 0.0, 1.0)

    if bundle.method == "platt" and bundle.platt is not None:
        raw = np.clip(np.asarray(raw_prob, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
        return bundle.platt.predict_proba(raw.reshape(-1, 1))[:, 1].reshape(np.asarray(raw_prob).shape)

    if bundle.method == "isotonic" and bundle.isotonic is not None:
        flat = raw_prob.ravel()
        out = bundle.isotonic.predict(flat)
        return out.reshape(raw_prob.shape)

    return np.clip(raw_prob, 0.0, 1.0)


def make_calibrated_predict_fn(
    base_predict_proba_1d: Callable[[np.ndarray], np.ndarray],
    bundle: CalibratorBundle,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap base model predict_proba[:,1] through calibration."""

    def _pred(X: np.ndarray) -> np.ndarray:
        raw = base_predict_proba_1d(X)
        return apply_calibration(bundle, raw)

    return _pred
