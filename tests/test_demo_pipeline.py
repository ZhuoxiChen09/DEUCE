from pathlib import Path

import pytest

from deuce.cli.main import main
from deuce.demo.artifacts import discover_demo_artifacts, validate_demo_artifacts
from deuce.demo.dataset import summarize_demo_dataset
from deuce.player_db import DEFAULT_DB_PATH, load_db
from deuce.presentation.demo_helpers import deterministic_explanation
from deuce.tournament_catalog import get_tournament_context, load_catalog
from deuce.training.features import build_player_stats_from_datasets
from deuce.training.predict import load_model, predict_match_probability


ROOT = Path(__file__).resolve().parents[1]


def test_demo_artifact_discovery_paths() -> None:
    artifacts = discover_demo_artifacts(ROOT)

    assert artifacts.build_command == "python -m deuce.cli demo build"
    assert artifacts.artifacts_dir == ROOT / "artifacts" / "demo"
    assert artifacts.model_path.name == "tennis_model_core.pkl"
    assert isinstance(artifacts.status_message(), str)


def test_demo_dataset_summary() -> None:
    summary = summarize_demo_dataset(ROOT / "data" / "demo" / "demo_matches.csv")

    assert summary["row_count"] == 1232
    assert summary["date_min"] == "20180101"
    assert summary["date_max"] == "20251222"
    assert set(summary["surfaces"]) == {"Clay", "Grass", "Hard"}


def test_demo_cli_help_runs(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["demo", "build", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Build local demo model and walk-forward artifacts" in out


def test_committed_demo_bundle_is_deployment_ready() -> None:
    summary = validate_demo_artifacts(ROOT)

    assert summary["dataset_rows"] == 1232
    assert summary["oos_predictions"] > 1000
    assert summary["model_features"] > 0
    assert summary["checksums_verified"] >= 6
    assert len(summary["feature_schema"]) == 64


def test_demo_validate_command(capsys) -> None:
    assert main(["demo", "validate"]) == 0
    assert "Demo runtime bundle is ready." in capsys.readouterr().out


def test_default_prediction_does_not_present_missing_seed_as_observed() -> None:
    artifacts = discover_demo_artifacts(ROOT)
    model, feature_names, medians = load_model(str(artifacts.model_path))
    player_db = load_db(str(ROOT / DEFAULT_DB_PATH))
    catalog = load_catalog(str(ROOT / "data" / "tournament_catalog.json"))
    stats_a = build_player_stats_from_datasets("Carlos Alcaraz", player_db=player_db)
    stats_b = build_player_stats_from_datasets("Jannik Sinner", player_db=player_db)
    context = get_tournament_context("BNP Paribas Open", catalog)
    result = predict_match_probability(
        stats_a,
        stats_b,
        model_core=model,
        feature_names=feature_names,
        medians=medians,
        tournament_name="BNP Paribas Open",
        catalog=catalog,
    )
    explanation = deterministic_explanation(
        player_a="Carlos Alcaraz",
        player_b="Jannik Sinner",
        probability_a=result.model_prob_a,
        contributions=result.per_feature_contributions,
        warnings=result.warnings,
        feature_values=result.feature_values,
        unavailable_features=set(
            result.diagnostics.get("unavailable_explanation_features") or []
        ),
    )

    assert context is not None
    assert result.feature_values["seed_diff_missing"] == 1.0
    assert not any("seed" in name for name in result.top_contributions)
    assert "seed" not in explanation.lower()
