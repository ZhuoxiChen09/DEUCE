"""Exercise artifact byte integrity across real builds and Git conversions."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from deuce.demo.artifacts import discover_demo_artifacts, validate_demo_artifacts
from deuce.demo.build import build_demo_artifacts
from deuce.training.predict import load_model
from deuce.training.train import load_csv_auto


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo-build")
    for relative in (
        ".gitattributes",
        "data/demo/demo_matches.csv",
        "data/tournament_catalog.json",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    build_demo_artifacts(root=root)
    validate_demo_artifacts(root)
    return root


@pytest.mark.parametrize("autocrlf", ["true", "input", "false"])
def test_built_bundle_survives_git_round_trip(built_demo, tmp_path, autocrlf):
    source = tmp_path / "source"
    shutil.copytree(built_demo, source)
    artifacts = discover_demo_artifacts(source)
    paths = (*artifacts.required_paths, source / "data/demo/demo_matches.csv")
    before = {path.relative_to(source): path.read_bytes() for path in paths}
    for relative, content in before.items():
        if relative.suffix in {".json", ".csv"}:
            content.decode("utf-8")
            assert b"\r" not in content, relative
            assert not content.startswith(b"\xef\xbb\xbf"), relative

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"core.autocrlf={autocrlf}",
             "-c", "core.safecrlf=false", *args],
            cwd=source,
        )

    git("init", "--quiet")
    git("add", ".gitattributes", "artifacts/demo", "data/demo/demo_matches.csv")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git("checkout-index", "--all", f"--prefix={checkout.as_posix()}/")
    for relative, content in before.items():
        # Check index (Git blob) bytes and fresh checkout bytes, including pickle.
        assert git("show", f":{relative.as_posix()}") == content
        assert (checkout / relative).read_bytes() == content
    validate_demo_artifacts(checkout)


def test_repeated_build_preserves_semantics(built_demo, tmp_path):
    second = tmp_path / "second-build"
    shutil.copytree(built_demo, second)
    build_demo_artifacts(root=second)
    validate_demo_artifacts(second)

    first_artifacts = discover_demo_artifacts(built_demo)
    second_artifacts = discover_demo_artifacts(second)
    first_manifest = json.loads(first_artifacts.manifest_path.read_text())
    second_manifest = json.loads(second_artifacts.manifest_path.read_text())
    for relative in first_manifest["checksums"]:
        if not relative.endswith(".pkl"):
            assert (built_demo / relative).read_bytes() == (second / relative).read_bytes()
    # Serialization bytes are not the model's semantic reproducibility contract.
    model_relative = first_artifacts.model_path.relative_to(built_demo).as_posix()
    del first_manifest["checksums"][model_relative]
    del second_manifest["checksums"][model_relative]
    assert first_manifest == second_manifest

    rows, _, _ = load_csv_auto(
        str(built_demo / "data/demo/demo_matches.csv"), None,
        feature_set="core", mirror_rows=False,
    )
    features = np.asarray([row["feature_vector"] for row in rows])
    probabilities = []
    for artifacts in (first_artifacts, second_artifacts):
        model, _, _ = load_model(str(artifacts.model_path))
        probabilities.append(
            model["calibrated_model"].predict_proba(model["scaler"].transform(features))
        )
    np.testing.assert_array_equal(*probabilities)

    metadata = [json.loads(path.read_text()) for path in (
        first_artifacts.build_metadata_path, second_artifacts.build_metadata_path,
    )]
    assert all(item["volatile"] is True for item in metadata)
    assert metadata[0]["generated_at"] != metadata[1]["generated_at"]
    assert "artifacts/demo/build_metadata.json" not in first_manifest["checksums"]


@pytest.mark.parametrize("mutation", ["crlf", "json-whitespace", "pickle-byte"])
def test_bundle_checksum_rejects_byte_changes(built_demo, tmp_path, mutation):
    root = tmp_path / "tampered"
    shutil.copytree(built_demo, root)
    artifacts = discover_demo_artifacts(root)
    path = artifacts.model_path if mutation == "pickle-byte" else artifacts.features_path
    content = path.read_bytes()
    path.write_bytes(
        content.replace(b"\n", b"\r\n") if mutation == "crlf" else content + b" "
    )
    with pytest.raises(ValueError, match="Checksum mismatch for"):
        validate_demo_artifacts(root)
