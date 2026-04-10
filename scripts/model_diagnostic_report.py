"""
Generate diagnostic stats for the tennis prediction model (Core).
Run from project root: python scripts/model_diagnostic_report.py
Outputs JSON to stdout for use in MODEL_DIAGNOSTIC_REPORT.md.
"""
import json
import sys
from pathlib import Path

# Run from kalshi_prediction_market
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        log_loss,
        roc_auc_score,
        confusion_matrix,
        brier_score_loss,
    )

    from model_train import load_historical_csv
    from model_predict import load_model
    from model_features import CORE_FEATURE_NAMES

    data_path = ROOT / "data" / "historical_2026_v3.csv"
    model_path = ROOT / "artifacts" / "tennis_model_core.pkl"

    if not data_path.exists():
        print(json.dumps({"error": f"Data not found: {data_path}"}), file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(json.dumps({"error": f"Model not found: {model_path}"}), file=sys.stderr)
        sys.exit(1)

    rows, labels, tournaments = load_historical_csv(
        str(data_path), catalog=None, feature_set="core", require_atp_coverage_for_enhanced=False
    )
    X = np.array([r["feature_vector"] for r in rows], dtype=float)
    y = np.array(labels, dtype=int)
    n = len(y)

    bundle, feature_names, _ = load_model(str(model_path))
    model = bundle["model"]
    coef = model.coef_[0]
    scaler = bundle.get("scaler")
    calibrated = bundle.get("calibrated_model")

    val_frac = 0.2
    random_state = 42
    from collections import Counter
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    use_split = val_frac > 0 and val_frac < 1 and n >= 4
    counts = Counter(int(v) for v in y)
    if use_split and min(counts.values()) < 2:
        use_split = False
    if use_split:
        i_train, i_val = train_test_split(idx, test_size=val_frac, random_state=random_state, stratify=y)
    else:
        i_train, i_val = idx, idx

    X_train, X_val = X[i_train], X[i_val]
    y_train, y_val = y[i_train], y[i_val]

    acc_train = acc_val = ll_train = ll_val = brier_val = None
    auc_val = None
    cm = None

    if scaler is not None and calibrated is not None:
        try:
            X_train_s = scaler.transform(X_train)
            X_val_s = scaler.transform(X_val)
            prob_train = calibrated.predict_proba(X_train_s)[:, 1]
            prob_val = calibrated.predict_proba(X_val_s)[:, 1]
            pred_train = (prob_train >= 0.5).astype(int)
            pred_val = (prob_val >= 0.5).astype(int)
            acc_train = float(accuracy_score(y_train, pred_train))
            acc_val = float(accuracy_score(y_val, pred_val))
            ll_train = float(log_loss(y_train, np.clip(prob_train, 1e-15, 1 - 1e-15)))
            ll_val = float(log_loss(y_val, np.clip(prob_val, 1e-15, 1 - 1e-15)))
            brier_val = float(brier_score_loss(y_val, prob_val))
            auc_val = float(roc_auc_score(y_val, prob_val))
            cm = confusion_matrix(y_val, pred_val).tolist()
        except Exception as e:
            pass  # e.g. sklearn version mismatch with pickled model

    # Feature missing: for each Core feature, if it has a _missing counterpart, % missing = mean(missing_flag)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    missing_stats = []
    for i, name in enumerate(feature_names):
        if name.endswith("_missing"):
            base = name.replace("_missing", "")
            num_missing = int(np.sum(X[:, i] == 1))
            pct = 100.0 * num_missing / n if n else 0
            missing_stats.append({
                "feature_name": base,
                "percent_missing": round(pct, 2),
                "num_missing": num_missing,
                "num_total": n,
            })
    missing_stats.sort(key=lambda x: -x["percent_missing"])

    # Coefficient table (by absolute value)
    coef_list = [{"feature_name": feature_names[i], "coefficient": round(float(coef[i]), 6)} for i in range(len(feature_names))]
    coef_list.sort(key=lambda x: -abs(x["coefficient"]))
    for r in range(len(coef_list)):
        coef_list[r]["rank"] = r + 1

    # Date range from CSV
    import csv
    dates = []
    with open(data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = (row.get("date") or "").strip()
            if d and len(d) >= 8:
                dates.append(d[:8])
    date_min = min(dates) if dates else "N/A"
    date_max = max(dates) if dates else "N/A"

    out = {
        "n_total": n,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "date_range": [date_min, date_max],
        "train_accuracy": acc_train,
        "val_accuracy": acc_val,
        "train_log_loss": ll_train,
        "val_log_loss": ll_val,
        "val_brier": brier_val,
        "val_roc_auc": auc_val,
        "confusion_matrix_val": cm,
        "missing_data_table": missing_stats,
        "coefficient_table": coef_list,
        "feature_names": feature_names,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
