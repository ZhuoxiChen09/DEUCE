# DEUCE Release-Candidate Checklist

## Completed release criteria

- [x] Missing source inputs are not presented as observed explanation factors.
- [x] Valid populated factors still appear with deterministic ordering.
- [x] Walk-forward boundaries use whole calendar-date groups.
- [x] Every fold has strictly later OOS dates than calibration dates.
- [x] Evaluation artifacts were regenerated from the corrected pipeline.
- [x] Semantic artifact metadata contains repository-relative paths only.
- [x] Volatile timestamp/host metadata is isolated and explicitly labeled.
- [x] Manifest validation covers files, schemas, versions, row counts, checksums,
      metrics, and fold date windows.
- [x] README documents the supported Streamlit demo and de-scopes legacy systems.
- [x] Altair is declared as a direct dependency.
- [x] Python 3.11 matches `runtime.txt`.
- [x] Offline operation requires no API keys or providers.

## Files to stage

Use `git status --short` as the final authority. The intended release set is:

```text
.env.example
.gitignore
.streamlit/config.toml
THIRD_PARTY_NOTICES.md
docs/demo/DEMO_SCRIPT.md
docs/project/PROJECT_STATUS_AUDIT.md
README.md
docs/release/RELEASE_CANDIDATE_CHECKLIST.md
app.py
runtime.txt
requirements.txt
legacy_app.py
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
tests/test_demo_helpers.py
tests/test_demo_pipeline.py
tests/test_walk_forward.py
artifacts/demo/artifact_manifest.json
artifacts/demo/build_metadata.json
artifacts/demo/demo_metrics.json
artifacts/demo/oos_predictions.csv
artifacts/demo/tennis_model_core.pkl
artifacts/demo/tennis_model_core.pkl.features.json
artifacts/demo/walk_forward_report.json
```

Do not stage caches, local databases, logs, secrets, virtual environments, or
unrelated large artifacts.

## Private-only release gate

This release candidate may be committed and pushed only to a private remote
until demo data and derived artifact redistribution rights are cleared. Do not
make the repository public, publish the bundled data/artifacts, or deploy the
demo publicly before that clearance exists.

## Clean-clone commands

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m deuce.cli demo build
python -m deuce.cli demo validate
python -m deuce.cli feature-tiers
python -m py_compile app.py deuce/presentation/demo_app.py deuce/presentation/demo_helpers.py
git diff --check
python -m streamlit run app.py
```

## Manual smoke-test checklist

- [x] Overview opens and displays 1,232 dataset rows and updated metrics.
- [x] Match Predictor opens.
- [x] Default Carlos Alcaraz vs. Jannik Sinner prediction renders.
- [x] Explanation and strongest-factor chart do not present missing seed, entry,
      home, or match-progression fields as observed signals.
- [x] Backtest Explorer displays 1,072 OOS rows and updated metrics.
- [x] Walk-forward fold table renders.
- [x] Methodology states that folds split on whole dates and OOS dates are
      strictly later than calibration dates.
- [x] Browser console has no application-breaking errors.

Last verified locally on June 26, 2026. The browser console emitted Vega/Altair
chart warnings, but no page-breaking errors were observed.

## Unresolved risks

- Dataset source attribution and redistribution permission need release-owner confirmation.
- Public repository distribution and public deployment are blocked until that
  clearance exists.
- Pickled scikit-learn models remain Python/package-version sensitive.
- Player context is a static local snapshot and may become stale.
- Legacy API/provider/market modules are retained but are not release-supported.
- No public deployment or production monitoring has been verified.

## Deployment prerequisites

- Push the complete intended release set only to a confirmed-private Git branch.
- Use Python 3.11.
- Configure `app.py` as the Streamlit entrypoint.
- Do not configure secrets; the supported demo requires none.
- Do not deploy publicly until demo data and artifact redistribution rights are
  cleared.
