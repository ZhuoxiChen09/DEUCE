"""Build reproducible local artifacts for the Streamlit portfolio demo."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deuce.backtest.engine import BacktestConfig
from deuce.demo.artifacts import (
    ARTIFACT_MANIFEST_VERSION,
    BUILD_COMMAND,
    DemoArtifacts,
    discover_demo_artifacts,
    sha256_file,
)
from deuce.demo.dataset import DEMO_DATASET_NAME, build_demo_dataset, summarize_demo_dataset
from deuce.research.walk_forward import (
    aggregate_report_to_json,
    run_walk_forward_from_csv,
    save_walk_forward_outputs,
)
from deuce.training.train import train_and_save


def _repo_root(root: Path | str | None = None) -> Path:
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parents[2]


def _repo_relative(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _safe_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _confidence_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20):
        subset = [r for r in rows if abs(float(r["model_prob_a"]) - 0.5) >= threshold]
        if not subset:
            out.append({"threshold": threshold, "n": 0, "accuracy": None})
            continue
        correct = sum(int(r["predicted_label"] == r["outcome_a_won"]) for r in subset)
        out.append({"threshold": threshold, "n": len(subset), "accuracy": correct / len(subset)})
    return out


def _calibration_bins(rows: list[dict[str, Any]], *, n_bins: int = 5) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        subset = [
            r
            for r in rows
            if lo <= float(r["model_prob_a"]) < hi or (i == n_bins - 1 and float(r["model_prob_a"]) == 1.0)
        ]
        if not subset:
            bins.append({"bin_min": lo, "bin_max": hi, "n": 0, "avg_predicted": None, "observed_rate": None})
            continue
        bins.append(
            {
                "bin_min": lo,
                "bin_max": hi,
                "n": len(subset),
                "avg_predicted": sum(float(r["model_prob_a"]) for r in subset) / len(subset),
                "observed_rate": sum(int(r["outcome_a_won"]) for r in subset) / len(subset),
            }
        )
    return bins


def _write_oos_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "fold_index",
        "model_prob_a",
        "outcome_a_won",
        "predicted_label",
        "correct",
        "confidence_margin",
        "cumulative_accuracy",
    ]
    correct_count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            p = float(row["model_prob_a"])
            y = int(row["outcome_a_won"])
            pred = 1 if p >= 0.5 else 0
            correct = int(pred == y)
            correct_count += correct
            writer.writerow(
                {
                    "row_index": idx,
                    "fold_index": row.get("fold_index", ""),
                    "model_prob_a": f"{p:.12f}",
                    "outcome_a_won": y,
                    "predicted_label": pred,
                    "correct": correct,
                    "confidence_margin": f"{abs(p - 0.5):.12f}",
                    "cumulative_accuracy": f"{correct_count / idx:.12f}",
                }
            )


def _demo_metrics(report_json: dict[str, Any], oos_rows: list[dict[str, Any]]) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for row in oos_rows:
        p = float(row["model_prob_a"])
        y = int(row["outcome_a_won"])
        pred = 1 if p >= 0.5 else 0
        enriched.append({**row, "predicted_label": pred, "correct": int(pred == y)})

    n = len(enriched)
    accuracy = (sum(int(r["correct"]) for r in enriched) / n) if n else 0.0
    y_true = [int(r["outcome_a_won"]) for r in enriched]
    y_score = [float(r["model_prob_a"]) for r in enriched]
    return {
        "n_oos_predictions": n,
        "accuracy": accuracy,
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "log_loss": report_json.get("log_loss"),
        "brier": report_json.get("brier"),
        "confidence_thresholds": _confidence_thresholds(enriched),
        "calibration_bins": _calibration_bins(enriched),
    }


def build_demo_artifacts(
    *,
    root: Path | str | None = None,
    rebuild_dataset: bool = False,
) -> dict[str, Any]:
    """Build the demo dataset if needed, train Core, and write report artifacts."""
    repo = _repo_root(root)
    source_csv = repo / "data" / "historical_tml_2000_2025.csv"
    data_path = repo / "data" / "demo" / DEMO_DATASET_NAME
    artifacts: DemoArtifacts = discover_demo_artifacts(repo)
    catalog_path = repo / "data" / "tournament_catalog.json"

    if rebuild_dataset or not data_path.exists():
        dataset_summary = build_demo_dataset(source_csv, data_path)
    else:
        dataset_summary = summarize_demo_dataset(data_path)
        dataset_summary["source_csv"] = str(source_csv)

    artifacts.artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_summary = train_and_save(
        str(data_path),
        str(artifacts.model_path),
        model_type="core",
        split_strategy="time",
        calibration_method="platt",
        test_frac=0.10,
        val_frac=0.20,
        random_state=42,
        catalog_path=str(catalog_path),
        data_reference=_repo_relative(data_path, repo),
        train_file_references=[_repo_relative(data_path, repo)],
    )

    report = run_walk_forward_from_csv(
        str(data_path),
        model_type="core",
        catalog_path=str(catalog_path),
        mirror_rows=False,
        mode="expanding",
        val_frac=0.20,
        calibration_method="platt",
        random_state=42,
        min_pool_rows=160,
        n_oos_folds=5,
        backtest_cfg=BacktestConfig(edge_threshold=0.0),
    )
    save_walk_forward_outputs(report, out_json=str(artifacts.report_path))
    report_json = aggregate_report_to_json(report)
    metrics = _demo_metrics(report_json, report.oos_rows)
    _write_oos_predictions(artifacts.oos_predictions_path, report.oos_rows)

    metrics_payload = dict(metrics)
    artifacts.metrics_path.write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8", newline="\n"
    )

    from deuce.training.predict import load_model

    model_bundle, _, _ = load_model(str(artifacts.model_path))
    artifacts.build_metadata_path.write_text(
        json.dumps(
            {
                "volatile": True,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "command": BUILD_COMMAND,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "executable_name": Path(sys.executable).name,
            },
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )

    semantic_paths = (
        data_path,
        artifacts.model_path,
        artifacts.features_path,
        artifacts.report_path,
        artifacts.oos_predictions_path,
        artifacts.metrics_path,
    )
    manifest = {
        "artifact_manifest_version": ARTIFACT_MANIFEST_VERSION,
        "command": BUILD_COMMAND,
        "reproducibility": {
            "standard": "semantic outputs, metrics, and schemas are deterministic within a fixed numerical environment; checksums verify exact bundle bytes",
            "volatile_metadata": _repo_relative(artifacts.build_metadata_path, repo),
            "pickle_bytes_required_identical": False,
        },
        "expected_files": [
            _repo_relative(path, repo) for path in artifacts.required_paths
        ],
        "checksums": {
            _repo_relative(path, repo): sha256_file(path) for path in semantic_paths
        },
        "dataset": dataset_summary,
        "model": {
            "path": _repo_relative(artifacts.model_path, repo),
            "features_path": _repo_relative(artifacts.features_path, repo),
            "type": "core",
            "bundle_version": model_bundle.get("version"),
            "artifact_schema_version": model_bundle.get("artifact_schema_version"),
            "code_version": model_bundle.get("code_version"),
            "model_class": model_bundle.get("model_class"),
            "sklearn_version": model_bundle.get("sklearn_version"),
            "feature_schema": model_bundle.get("feature_schema"),
            "training_summary": train_summary,
        },
        "walk_forward": {
            "report_path": _repo_relative(artifacts.report_path, repo),
            "oos_predictions_path": _repo_relative(artifacts.oos_predictions_path, repo),
            "metrics_path": _repo_relative(artifacts.metrics_path, repo),
            "mode": report_json.get("mode"),
            "n_folds": report_json.get("n_folds"),
            "n_oos_predictions": report_json.get("n_oos_predictions"),
        },
        "metrics": metrics,
        "notes": [
            "Model training reuses the existing leakage-aware Core feature set.",
            "Walk-forward evaluation uses original demo rows without mirrored augmentation.",
            "All train, calibration, and OOS boundaries use whole calendar-date groups.",
            "No live scraping or external API calls are required.",
        ],
    }
    manifest["dataset"]["source_csv"] = _repo_relative(source_csv, repo)
    manifest["dataset"]["output_csv"] = _repo_relative(data_path, repo)
    artifacts.manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    return manifest


def format_build_summary(manifest: dict[str, Any]) -> str:
    """Format concise CLI output after a successful demo build."""
    dataset = manifest["dataset"]
    model = manifest["model"]
    wf = manifest["walk_forward"]
    metrics = manifest["metrics"]
    auc = metrics.get("roc_auc")
    auc_text = "n/a" if auc is None else f"{auc:.4f}"
    lines = [
        "Demo build complete.",
        f"Dataset: {dataset['output_csv']} ({dataset['row_count']} rows, {dataset['date_min']} to {dataset['date_max']})",
        f"Model: {model['path']}",
        f"Walk-forward report: {wf['report_path']}",
        f"OOS predictions: {wf['oos_predictions_path']}",
        f"Metrics: {manifest['walk_forward']['metrics_path']}",
        f"Manifest: {Path(model['path']).parent / 'artifact_manifest.json'}",
        (
            "Scores: "
            f"n={metrics['n_oos_predictions']}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"ROC-AUC={auc_text}, "
            f"Brier={float(metrics['brier']):.4f}, "
            f"log_loss={float(metrics['log_loss']):.4f}"
        ),
    ]
    return "\n".join(lines)
