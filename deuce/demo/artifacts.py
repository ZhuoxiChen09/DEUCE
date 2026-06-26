"""Discovery and validation helpers for portfolio-demo artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


BUILD_COMMAND = "python -m deuce.cli demo build"
ARTIFACT_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class DemoArtifacts:
    """Resolved demo artifact paths and readiness status."""

    root: Path
    artifacts_dir: Path
    model_path: Path
    features_path: Path
    report_path: Path
    oos_predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    build_metadata_path: Path
    build_command: str = BUILD_COMMAND

    @property
    def required_paths(self) -> tuple[Path, ...]:
        return (
            self.model_path,
            self.features_path,
            self.report_path,
            self.oos_predictions_path,
            self.metrics_path,
            self.manifest_path,
            self.build_metadata_path,
        )

    @property
    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(path for path in self.required_paths if not path.exists())

    @property
    def ready(self) -> bool:
        return not self.missing_paths

    def status_message(self) -> str:
        """Return a concise human-facing readiness or remediation message."""
        if self.ready:
            return f"Demo artifacts found in {self.artifacts_dir}"
        missing = ", ".join(path.name for path in self.missing_paths)
        return (
            f"Demo artifacts are missing ({missing}). "
            f"Run `{self.build_command}` from the repository root."
        )


def _repo_root(start: Path | None = None) -> Path:
    if start is not None:
        return start.resolve()
    return Path(__file__).resolve().parents[2]


def discover_demo_artifacts(root: Path | str | None = None) -> DemoArtifacts:
    """Resolve the standard local demo artifact bundle under ``artifacts/demo``."""
    repo_root = _repo_root(Path(root) if root is not None else None)
    artifacts_dir = repo_root / "artifacts" / "demo"
    model_path = artifacts_dir / "tennis_model_core.pkl"
    return DemoArtifacts(
        root=repo_root,
        artifacts_dir=artifacts_dir,
        model_path=model_path,
        features_path=model_path.with_suffix(model_path.suffix + ".features.json"),
        report_path=artifacts_dir / "walk_forward_report.json",
        oos_predictions_path=artifacts_dir / "oos_predictions.csv",
        metrics_path=artifacts_dir / "demo_metrics.json",
        manifest_path=artifacts_dir / "artifact_manifest.json",
        build_metadata_path=artifacts_dir / "build_metadata.json",
    )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(label: str, left: Any, right: Any, *, tolerance: float = 1e-10) -> None:
    if left is None and right is None:
        return
    if left is None or right is None or not math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    ):
        raise ValueError(f"{label} mismatch: {left!r} != {right!r}")


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if not isinstance(value, str):
        return False
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith(("/", "\\\\"))


def validate_demo_artifacts(root: Path | str | None = None) -> dict[str, Any]:
    """Validate schemas, checksums, row counts, metrics, and temporal integrity."""
    artifacts = discover_demo_artifacts(root)
    if not artifacts.ready:
        missing = ", ".join(
            str(path.relative_to(artifacts.root)) for path in artifacts.missing_paths
        )
        raise FileNotFoundError(f"Missing demo runtime files: {missing}")

    from deuce.training.predict import load_model

    model, feature_names, medians = load_model(str(artifacts.model_path))
    if not isinstance(model, dict) or not feature_names or not medians:
        raise ValueError("Demo model bundle is incomplete or unversioned.")

    metrics = json.loads(artifacts.metrics_path.read_text(encoding="utf-8"))
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    features = json.loads(artifacts.features_path.read_text(encoding="utf-8"))
    build_metadata = json.loads(
        artifacts.build_metadata_path.read_text(encoding="utf-8")
    )

    if int(manifest.get("artifact_manifest_version") or 0) != ARTIFACT_MANIFEST_VERSION:
        raise ValueError("Unsupported demo artifact manifest version.")
    if _contains_absolute_path(manifest) or _contains_absolute_path(features):
        raise ValueError("Semantic artifact metadata contains an absolute local path.")
    model_metadata = {
        key: model.get(key)
        for key in (
            "train_data_path",
            "train_files",
            "feature_schema",
            "code_version",
            "sklearn_version",
        )
    }
    if _contains_absolute_path(model_metadata):
        raise ValueError("Model bundle metadata contains an absolute local path.")
    if not build_metadata.get("volatile"):
        raise ValueError("Build metadata must be explicitly marked volatile.")

    expected_files = set(manifest.get("expected_files") or [])
    runtime_files = {
        path.relative_to(artifacts.root).as_posix() for path in artifacts.required_paths
    }
    if expected_files != runtime_files:
        raise ValueError("Manifest expected-file list does not match the runtime bundle.")

    checksums = manifest.get("checksums") or {}
    if not checksums:
        raise ValueError("Manifest does not contain semantic artifact checksums.")
    for relative_path, expected_digest in checksums.items():
        artifact_path = artifacts.root / relative_path
        if not artifact_path.exists():
            raise FileNotFoundError(f"Checksummed artifact is missing: {relative_path}")
        if sha256_file(artifact_path) != expected_digest:
            raise ValueError(f"Checksum mismatch for {relative_path}")

    feature_schema = manifest.get("model", {}).get("feature_schema") or {}
    if feature_schema != model.get("feature_schema") or feature_schema != features.get(
        "feature_schema"
    ):
        raise ValueError("Feature-schema metadata differs across artifact files.")
    if int(feature_schema.get("n_features") or 0) != len(feature_names):
        raise ValueError("Feature-schema count does not match the loaded model.")
    if features.get("feature_names") != feature_names:
        raise ValueError("Feature order differs between model bundle and sidecar.")
    if manifest.get("model", {}).get("bundle_version") != model.get("version"):
        raise ValueError("Model bundle version does not match the manifest.")
    if manifest.get("model", {}).get("code_version") != model.get("code_version"):
        raise ValueError("Model code version does not match the manifest.")

    with artifacts.oos_predictions_path.open("r", encoding="utf-8", newline="") as f:
        oos_rows = list(csv.DictReader(f))
    expected_oos = int(metrics.get("n_oos_predictions") or 0)
    if expected_oos <= 0 or len(oos_rows) != expected_oos:
        raise ValueError(
            f"OOS prediction count mismatch: metrics={expected_oos}, csv={len(oos_rows)}"
        )
    if int(report.get("n_oos_predictions") or 0) != expected_oos:
        raise ValueError("Walk-forward report does not match demo metrics.")

    dataset_path = artifacts.root / manifest["dataset"]["output_csv"]
    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        dataset_count = sum(1 for _ in csv.DictReader(f))
    if dataset_count != int(manifest["dataset"]["row_count"]):
        raise ValueError("Dataset row count does not match the manifest.")

    y_true = [int(float(row["outcome_a_won"])) for row in oos_rows]
    y_score = [float(row["model_prob_a"]) for row in oos_rows]
    predictions = [1 if score >= 0.5 else 0 for score in y_score]
    accuracy = sum(
        int(pred == actual) for pred, actual in zip(predictions, y_true)
    ) / expected_oos
    brier = sum(
        (score - actual) ** 2 for score, actual in zip(y_score, y_true)
    ) / expected_oos
    log_loss = -sum(
        actual * math.log(min(max(score, 1e-15), 1.0 - 1e-15))
        + (1 - actual)
        * math.log(min(max(1.0 - score, 1e-15), 1.0 - 1e-15))
        for score, actual in zip(y_score, y_true)
    ) / expected_oos
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc_auc = None

    _assert_close("accuracy", metrics.get("accuracy"), accuracy)
    _assert_close("Brier score", metrics.get("brier"), brier)
    _assert_close("log loss", metrics.get("log_loss"), log_loss)
    _assert_close("ROC-AUC", metrics.get("roc_auc"), roc_auc)
    _assert_close("report Brier score", report.get("brier"), metrics.get("brier"))
    _assert_close("report log loss", report.get("log_loss"), metrics.get("log_loss"))
    for key in ("accuracy", "roc_auc", "brier", "log_loss"):
        _assert_close(
            f"manifest {key}",
            manifest.get("metrics", {}).get(key),
            metrics.get(key),
        )

    for fold in report.get("folds") or []:
        train_max = (fold.get("train_date_window") or {}).get("max")
        cal_max = (fold.get("val_date_window") or {}).get("max")
        oos_min = (fold.get("oos_date_window") or {}).get("min")
        if train_max is None or cal_max is None or oos_min is None:
            raise ValueError("Fold date windows are incomplete.")
        if not (int(train_max) <= int(cal_max) < int(oos_min)):
            raise ValueError("Fold date windows do not satisfy strict temporal separation.")

    return {
        "artifacts_dir": artifacts.artifacts_dir.relative_to(artifacts.root).as_posix(),
        "model_features": len(feature_names),
        "oos_predictions": expected_oos,
        "accuracy": float(metrics["accuracy"]),
        "roc_auc": float(metrics["roc_auc"]),
        "brier": float(metrics["brier"]),
        "log_loss": float(metrics["log_loss"]),
        "dataset_rows": dataset_count,
        "feature_schema": feature_schema.get("sha256"),
        "checksums_verified": len(checksums),
    }
