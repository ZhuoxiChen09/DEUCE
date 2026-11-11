# Clean Clone Verification

> Historical verification dated June 26, 2026. Original paths and results below
> are preserved as recorded; they do not verify later commits or supersede CI.
> The status audit moved to [docs/archive](../archive/PROJECT_STATUS_AUDIT.md).
> See the [current roadmap](../roadmap.md).

## Verification Environment

* operating system: Microsoft Windows NT 10.0.26200.0
* Python version: Python 3.11.9
* virtual environment path/name: `<Python 3.11 virtual environment>` in `<temporary clean checkout>`
* Git commit or working-tree source used: local Git clone of `b63f5d7` plus exact overlay of the current 39-file release-candidate working-tree staging set from `docs/release/RELEASE_CANDIDATE_CHECKLIST.md`; no commit was created.
* verification date: June 26, 2026

## Repository Completeness

* required files present: yes. The clean verification checkout contained the full intended release set from `docs/release/RELEASE_CANDIDATE_CHECKLIST.md`.
* excluded files confirmed absent from the release set: yes. Caches, logs, local SQLite databases, secrets, and virtual environments are not part of the staging set. Verification-generated `.pytest_cache/`, `.deuce/`, `__pycache__/`, and Streamlit smoke-test logs were removed after verification. The clean verification `.venv/` remains only in the temporary checkout and is ignored.
* runtime artifacts present:
  * `artifacts/demo/artifact_manifest.json`
  * `artifacts/demo/build_metadata.json`
  * `artifacts/demo/demo_metrics.json`
  * `artifacts/demo/oos_predictions.csv`
  * `artifacts/demo/tennis_model_core.pkl`
  * `artifacts/demo/tennis_model_core.pkl.features.json`
  * `artifacts/demo/walk_forward_report.json`
* documentation files present:
  * `README.md`
  * `docs/project/PROJECT_STATUS_AUDIT.md`
  * `docs/release/RELEASE_CANDIDATE_CHECKLIST.md`
  * `docs/demo/DEMO_SCRIPT.md`
  * `docs/LAYOUT.md`
  * `docs/architecture.md`
  * `docs/development.md`
  * `data/demo/README.md`

## Commands and Results

| Command | Result | Concise output |
|---|---|---|
| `git status --short` | Pass | Current repo has exactly the intended 39 release files before this report was added. |
| `git diff --check` | Pass | No whitespace errors. |
| `git diff --stat` | Pass | 14 tracked files changed; untracked release files handled by checklist comparison. |
| `git diff --name-only` | Pass | Tracked changes are release-candidate source/docs/test/config updates. |
| `python --version` | Pass | Python 3.11.9 in fresh venv. |
| `python -m pip install --upgrade pip` | Pass | pip upgraded from 24.0 to 26.1.2 in fresh venv. |
| `python -m pip install -r requirements.txt` | Pass | Installed only from `requirements.txt`. Key packages: Streamlit 1.58.0, pandas 2.3.3, numpy 1.26.4, scikit-learn 1.5.2, Altair 5.5.0, pytest 9.1.1. |
| `python -m pip check` | Pass | `No broken requirements found.` |
| `python -m pytest tests -q` | Pass | `139 passed, 10 warnings in 29.30s`. |
| `python -m deuce.cli demo build` | Pass | Rebuilt 1,232-row demo dataset artifacts; OOS=1,072, accuracy=0.6157, ROC-AUC=0.6191, Brier=0.2419, log_loss=0.6779. |
| `python -m deuce.cli demo validate` | Pass | `Demo runtime bundle is ready. Scores: dataset=1232, OOS=1072, accuracy=0.6157, ROC-AUC=0.6191, Brier=0.2419`. |
| `python -m deuce.cli feature-tiers` | Pass | Printed Core, Enhanced, legacy, and leakage-risk feature manifests. |
| `python -m py_compile app.py deuce/presentation/demo_app.py deuce/presentation/demo_helpers.py` | Pass | No syntax errors. |
| `git diff --check` in clean verification checkout | Pass | No whitespace errors after applying the release-candidate working tree. |
| `python -m streamlit run app.py` | Pass | Clean-checkout app launched at `http://127.0.0.1:8501`. |

## Streamlit Smoke Test

* Overview: Pass. Page opened and displayed 1,232 demo matches, 1,072 OOS predictions, 61.6% accuracy, and ROC-AUC 0.619.
* Match Predictor: Pass. Page opened with Carlos Alcaraz vs. Jannik Sinner defaults.
* default Carlos Alcaraz vs. Jannik Sinner prediction: Pass. Prediction rendered with Carlos Alcaraz at 53.1% and Jannik Sinner at 46.9%.
* explanation integrity: Pass. The rendered explanation did not present missing seed fields as observed factors. The default explanation was: "The model favors Carlos Alcaraz over Jannik Sinner, led by ranking-points difference, Altitude factor, event tier. Interpret the result cautiously because some input data is incomplete."
* Backtest Explorer: Pass. Page rendered updated metrics and the Walk-Forward Folds table.
* Methodology: Pass. Page states that validation uses whole calendar dates and evaluates only on strictly later dates.

## Warnings or Non-Blocking Issues

* `pytest` emitted 10 warnings in the clean venv:
  * one Starlette/FastAPI TestClient warning that `httpx` with `starlette.testclient` is deprecated;
  * nine scikit-learn/SciPy L-BFGS-B `disp`/`iprint` deprecation warnings.
* Browser console emitted Vega/Altair warnings:
  * Vega-Lite spec version mismatch warnings;
  * infinite extent warnings for empty chart bins;
  * scale-binding/discrete-width warnings.
  These did not block page rendering or navigation.
* Git repeatedly warned that `<global Git ignore file>` was not accessible under the sandbox user. This did not affect repository status, diff checks, or clean verification.
* Dataset source attribution and redistribution permission remain unresolved and should be confirmed before public deployment.
* No public deployment was created or verified.

## Release Decision

`READY TO COMMIT, NOT READY TO DEPLOY`

The release candidate is ready to commit because a clean temp Git checkout with a fresh Python 3.11 virtual environment installed dependencies only from `requirements.txt`, passed all required commands, rebuilt/validated artifacts, and passed Streamlit smoke testing. It is not ready to deploy publicly until dataset provenance/redistribution permission is confirmed and a deployment target is configured and tested.

## Exact Files to Stage

Final staging list based on current `git status --short`:

```text
.env.example
.gitignore
.streamlit/config.toml
docs/release/CLEAN_CLONE_VERIFICATION.md
docs/demo/DEMO_SCRIPT.md
docs/project/PROJECT_STATUS_AUDIT.md
README.md
docs/release/RELEASE_CANDIDATE_CHECKLIST.md
app.py
artifacts/demo/artifact_manifest.json
artifacts/demo/build_metadata.json
artifacts/demo/demo_metrics.json
artifacts/demo/oos_predictions.csv
artifacts/demo/tennis_model_core.pkl
artifacts/demo/tennis_model_core.pkl.features.json
artifacts/demo/walk_forward_report.json
data/demo/README.md
data/demo/artifact_manifest.example.json
data/demo/demo_matches.csv
deuce/cli/main.py
deuce/demo/__init__.py
deuce/demo/artifacts.py
deuce/demo/build.py
deuce/demo/dataset.py
deuce/presentation/demo_app.py
deuce/presentation/demo_helpers.py
deuce/training/features.py
deuce/training/predict.py
deuce/training/temporal.py
deuce/training/train.py
deuce/training/walk_forward_folds.py
docs/LAYOUT.md
docs/architecture.md
docs/development.md
legacy_app.py
requirements.txt
runtime.txt
tests/test_demo_helpers.py
tests/test_demo_pipeline.py
tests/test_walk_forward.py
```
