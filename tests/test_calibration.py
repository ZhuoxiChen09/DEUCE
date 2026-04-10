import numpy as np

from deuce.training.calibration import apply_calibration, fit_calibration


def test_none_identity():
    raw = np.array([0.3, 0.7, 0.5])
    y = np.array([0, 1, 0])
    b = fit_calibration("none", raw, y)
    out = apply_calibration(b, raw)
    assert np.allclose(out, raw)


def test_platt_monotone():
    rng = np.random.RandomState(1)
    raw = rng.rand(30)
    y = (rng.rand(30) > 0.5).astype(int)
    b = fit_calibration("platt", raw, y, random_state=42)
    out = apply_calibration(b, raw)
    assert out.shape == raw.shape
    assert (out >= 0).all() and (out <= 1).all()
