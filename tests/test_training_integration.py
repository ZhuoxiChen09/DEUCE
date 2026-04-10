"""End-to-end training: time split, calibration, bundle reload, inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import model_train
from deuce.features.leakage import assert_no_leakage_in_feature_list
from deuce.training.temporal import assert_disjoint_train_val_test_dates, assert_strictly_ordered_dates
from model_features import CORE_FEATURE_NAMES
from model_predict import load_model


def _write_tiny_core_csv(path: Path, n_rows: int = 80) -> None:
    """Historical format: one row per match, increasing dates, minimal columns for core."""
    lines = [
        "player_a,player_b,winner,rank_a,rank_b,points_a,points_b,tourney_date,surface,round",
    ]
    base = 20240101
    for i in range(n_rows):
        d = base + i  # unique YYYYMMDD
        w = "A" if i % 2 == 0 else "B"
        lines.append(
            f"P{i%3}A,P{i%4}B,{w},{10+i%50},{20+i%50},{1000+i},{900+i},{d},Hard,R32"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def test_train_time_split_bundle_and_infer(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    out_pkl = tmp_path / "m_core.pkl"
    _write_tiny_core_csv(csv_path, n_rows=80)

    model_train.train_and_save(
        str(csv_path),
        str(out_pkl),
        model_type="core",
        mirror_rows=False,
        split_strategy="time",
        calibration_method="platt",
        test_frac=0.1,
        val_frac=0.2,
        random_state=42,
    )

    assert out_pkl.exists()

    bundle, _, _ = load_model(str(out_pkl))

    assert bundle.get("version") == 3
    assert bundle.get("model_tier") == "core"
    assert bundle.get("feature_set_name") == "core_prematch"
    cal = bundle.get("calibration") or {}
    assert cal.get("method") == "platt"
    assert "val_log_loss_calibrated" in cal

    assert_no_leakage_in_feature_list(CORE_FEATURE_NAMES, context="integration test")

    dates = np.array(
        [int(r.get("match_date") or 0) for r in model_train.load_csv_auto(str(csv_path), None, feature_set="core", mirror_rows=False)[0]],
        dtype=np.int64,
    )
    order = np.argsort(dates, kind="stable")
    assert_strictly_ordered_dates(dates, order, context="integration")

    n = len(dates)
    n_test = max(1, int(n * 0.1))
    i_test = order[-n_test:]
    pool = order[:-n_test]
    cut = int(len(pool) * (1 - 0.2))
    i_train = pool[:cut]
    i_val = pool[cut:]
    assert_disjoint_train_val_test_dates(dates, i_train, i_val, i_test)

    rows, labels, _ = model_train.load_csv_auto(str(csv_path), None, feature_set="core", mirror_rows=False)
    X = np.array([r["feature_vector"] for r in rows], dtype=float)
    y = np.array(labels, dtype=int)

    bundle2, _, _ = load_model(str(out_pkl))
    assert isinstance(bundle2, dict)
    cal_m = bundle2["calibrated_model"]
    scaler = bundle2["scaler"]
    prob = cal_m.predict_proba(scaler.transform(X[:3]))[:, 1]
    assert np.all((prob >= 0) & (prob <= 1))

    # Labels alignment smoke check
    assert len(y) == X.shape[0]
