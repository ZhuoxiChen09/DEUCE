# Project Status Audit

## Release-Candidate Update — June 26, 2026

The Release-Candidate Model Trust and Packaging milestone is implemented in the
working tree.

- Prediction explanations now pair factors with missingness/source availability.
  Missing seed, entry, event, demographic, home-context, and progression inputs
  are suppressed as observed factors; explicit missing-data effects remain
  separately labeled.
- Expanding and rolling walk-forward folds now split on whole calendar-date
  groups. Every generated fold satisfies
  `max(train_date) <= max(calibration_date) < min(oos_date)`, with no date shared
  across train, calibration, or OOS partitions.
- Artifact metadata uses repository-relative paths. Semantic files are
  timestamp-free and checksum-validated; volatile timestamp/host information is
  isolated in `artifacts/demo/build_metadata.json`.
- Validation now checks required files, feature-schema identity, model/code
  versions, dataset and OOS row counts, semantic checksums, cross-file metrics,
  and fold date windows.
- Documentation now treats the four-page offline Streamlit app as the supported
  product and labels API, provider, market, and legacy-app surfaces as
  experimental or archived.

Corrected walk-forward metrics:

| Metric | Previous | Release candidate |
|---|---:|---:|
| OOS predictions | 1,072 | 1,072 |
| Accuracy | 0.6129 | 0.6157 |
| ROC-AUC | 0.6255 | 0.6191 |
| Brier score | 0.2397 | 0.2419 |
| Log loss | 0.6728 | 0.6779 |

The metric changes are caused by moving row-level fold boundaries to whole-date
boundaries. No performance values were edited manually.

Remaining limitations include the static player-data snapshot, dataset
provenance/redistribution confirmation, version sensitivity of Python pickle
artifacts, no verified public deployment, and experimental legacy/API/provider
surfaces outside the supported demo.

Release-candidate verification was rerun on June 26, 2026 with Python 3.11.9:
`139 passed`, demo artifact build/validation passed, feature-tier CLI passed,
`py_compile` passed, `git diff --check` passed, and a local Streamlit smoke test
verified Overview, Match Predictor, Backtest Explorer, and Methodology. Browser
console output contained Vega/Altair warnings only; no application-breaking
errors were observed.

The sections below preserve the pre-release-candidate audit baseline. Where they
conflict with this update, the release-candidate section and regenerated
artifacts are authoritative.

## 1. Executive Summary

DEUCE currently has a functional local portfolio demo for tennis match prediction and walk-forward evaluation, backed by a real calibrated logistic-regression bundle and committed-source historical data in the working tree. The four-page Streamlit experience, artifact validation, model inference, walk-forward reports, FastAPI foundation, and 133 automated tests all work locally. However, the portfolio-demo implementation and runtime bundle are not committed to Git, no deployment URL or screenshots exist, the documented clean-clone experience therefore does not match `origin/main`, and there are model-trust issues in the explanation and temporal-validation logic. Overall completion is estimated at **68%**: the core local demo is substantially implemented, but release integrity, evaluation rigor, documentation alignment, and the larger legacy/API product scope remain incomplete.

## 2. Intended Product

The repository contains two overlapping intended products:

1. **Current default product:** a reproducible, recruiter-facing Streamlit tennis analytics demo.
   - Review the dataset and headline out-of-sample metrics.
   - Compare two players using a locally stored calibrated model.
   - Inspect deterministic feature explanations and confidence.
   - Explore walk-forward evaluation, calibration, confidence buckets, and fold metrics.
   - Review methodology, limitations, and reproduction commands.

2. **Retained legacy/research platform:** a broader tennis market and live-score toolkit.
   - Fetch live or scheduled tennis data from public/optional providers.
   - Ingest normalized match state into SQLite through FastAPI.
   - Inspect market pricing, edge, bankroll, and research backtests.
   - Run legacy Kalshi/ATP workflows through `cli.py` and `legacy_app.py`.

Primary architecture for the default demo:

`data/demo/demo_matches.csv` -> `deuce.training.features` -> calibrated logistic regression -> `deuce.research.walk_forward` -> `artifacts/demo/*` -> `deuce.presentation.demo_app` -> `app.py`

Supporting architecture retained in the repository:

`deuce.live_scores/providers` -> reconciliation/deduplication -> SQLAlchemy storage -> services -> FastAPI routes, plus separate market-pricing, edge, bankroll, research, and legacy Streamlit presentation modules.

## 3. Completed and Verified

| Area | What exists | Evidence / relevant files | Verification status |
|---|---|---|---|
| UI | Default four-page Streamlit app: Overview, Match Predictor, Backtest Explorer, Methodology | `app.py`, `deuce/presentation/demo_app.py`, `deuce/presentation/demo_helpers.py` | Verified |
| UI routing | Sidebar radio navigation opens all four pages; the default route is Overview | `deuce/presentation/demo_app.py` | Verified in browser |
| Match prediction | Loads the local Core model, builds player statistics from `data/player_stats.json`, applies tournament/surface context, and returns calibrated probabilities | `deuce/training/predict.py`, `deuce/training/features.py`, `deuce/player_db.py` | Verified with Carlos Alcaraz vs. Jannik Sinner |
| Explanations | Deterministic confidence labels, feature labels, contribution grouping, and explanation text | `deuce/presentation/demo_helpers.py`, `tests/test_demo_helpers.py` | Partially verified; integrity issue documented below |
| Demo data | Deterministic 1,232-row, 67-column slice spanning 2018-01-01 through 2025-12-22 across hard, clay, and grass | `data/demo/demo_matches.csv`, `deuce/demo/dataset.py`, `data/demo/README.md` | Verified |
| Data source pipeline | Historical source CSV and year-level ATP/Challenger files are present; builder scripts exist | `data/historical_tml_2000_2025.csv`, `data/*.csv`, `scripts/build_historical_from_tml.py` | Partially verified |
| Model artifact | Version 3 Core bundle with 57 pre-match features, scaler, logistic regression, Platt calibration, medians, and metadata | `artifacts/demo/tennis_model_core.pkl`, sidecar JSON | Verified |
| Leakage controls | Core and Enhanced feature lists exclude declared score-derived leakage features | `deuce/features/leakage.py`, `deuce/training/features.py`, `tests/test_leakage.py` | Verified |
| Evaluation | Five expanding walk-forward folds, 1,072 OOS predictions, calibration metrics, Brier score, log loss, and fold details | `deuce/research/walk_forward.py`, `artifacts/demo/walk_forward_report.json`, `artifacts/demo/oos_predictions.csv` | Partially verified; same-date boundary risk remains |
| Reported metrics | Accuracy 0.6129, ROC-AUC 0.6255, Brier 0.2397, log loss 0.6728 | `artifacts/demo/demo_metrics.json` | Verified |
| Reproducibility command | `python -m deuce.cli demo build` rebuilds model and evaluation artifacts without external providers | `deuce/demo/build.py`, `deuce/cli/main.py` | Partially verified |
| Runtime validation | `python -m deuce.cli demo validate` loads the model and checks required files and OOS row counts | `deuce/demo/artifacts.py` | Verified |
| Tests | Unit/integration tests cover domain scoring, providers, storage, pipeline, API, model features/training, leakage, walk-forward evaluation, backtesting, demo artifacts, and helpers | `tests/` | Verified: 133 passed |
| API foundation | Health, ingest, matches, provider health, simulation, and placeholder prediction routes | `deuce/api/`, `deuce/services/`, `deuce/storage/` | Verified through tests and local TestClient smoke checks |
| Backtest library | Market math, edge filters, sizing, bankroll path, ROI, drawdown, and summaries | `deuce/backtest/engine.py`, `tests/test_backtest.py` | Verified by automated tests |
| CI configuration | Python 3.11 setup, dependency installation, tests, and CLI smoke checks | `.github/workflows/ci.yml` | Not verified for the current working tree because changes are uncommitted |
| Deployment configuration | Python 3.11 runtime target, Streamlit configuration, dependency file, artifact allow-list in `.gitignore` | `runtime.txt`, `.streamlit/config.toml`, `requirements.txt`, `.gitignore` | Partially verified |
| Documentation | Portfolio README, demo walkthrough script, architecture, development, provider, model, dataset, and backtest notes | `README.md`, `docs/demo/DEMO_SCRIPT.md`, `docs/`, `data/demo/README.md` | Partially verified; several documents are stale or incomplete |

## 4. In Progress or Partially Implemented

### Portfolio-demo release packaging

- **Current state:** The local working tree contains the new app, demo data, runtime artifacts, deployment configuration, tests, and documentation.
- **What is missing:** None of those additions are committed. `main`, `origin/main`, and `origin/HEAD` all still point to the single April 9 initial commit.
- **Why it matters:** A clean clone or Streamlit deployment from Git does not receive the audited demo implementation or its artifacts.
- **Recommended next action:** Resolve the model-trust findings, then intentionally stage and commit the complete release set and verify it in a clean environment.

### Public deployment and visual documentation

- **Current state:** Local Streamlit startup and all four pages work.
- **What is missing:** The README still contains a deployment URL placeholder. Referenced files `docs/images/overview.png`, `match-predictor.png`, `backtest-explorer.png`, and `methodology.png` do not exist.
- **Why it matters:** The repository presents a “Live Demo” and “Product Walkthrough” that are visibly incomplete.
- **Recommended next action:** Deploy the release candidate, capture real screenshots, add the URL, and verify all links from GitHub.

### Prediction explanation integrity

- **Current state:** Explanations are deterministic and use model contributions.
- **What is missing:** Contribution filtering does not suppress features whose underlying input is missing or represents only a standardized baseline.
- **Why it matters:** For the default Alcaraz/Sinner example, raw seed inputs are missing, but the UI says the prediction is led by “seed status.” The top contribution values are dominated by `seeded_flag_a` and `seeded_flag_b` even though both raw values are zero and both missing flags are one.
- **Recommended next action:** Pair value features with their missing indicators, suppress unavailable factors from explanations, and add a regression test for the default prediction.

### Strict temporal evaluation

- **Current state:** Walk-forward folds are ordered by date and use disjoint row indices.
- **What is missing:** Splits do not group rows by calendar date. The temporal assertions allow equality between calibration maximum date and OOS minimum date.
- **Why it matters:** Multiple generated folds contain calibration and OOS matches on the same date, which weakens the claim that all evaluation matches are strictly later than calibration data.
- **Recommended next action:** Split on unique date groups and require `max(calibration_date) < min(oos_date)`, then rebuild and re-report metrics.

### Artifact reproducibility and validation depth

- **Current state:** Repeated builds produced identical OOS CSV, walk-forward report, and semantic metrics.
- **What is missing:** The model pickle and feature sidecar are not byte-identical because they embed build timestamps and absolute local Windows paths. Validation does not check hashes, feature schema identity, package compatibility, or semantic consistency of all headline metrics.
- **Why it matters:** “Reproducible artifacts” currently means reproducible model outputs, not identical artifacts. Absolute paths also create noisy, machine-specific metadata.
- **Recommended next action:** Store repository-relative paths, separate volatile build metadata, and add checksums/schema/version checks to the manifest validator.

### FastAPI prediction path

- **Current state:** The API, persistence, provider abstraction, and ingest flow exist.
- **What is missing:** `/predictions/live/demo` is explicitly a placeholder and never loads player stats or a trained model.
- **Why it matters:** The API architecture suggests prediction capability that is not available through the HTTP surface.
- **Recommended next action:** Either implement one real pre-match prediction endpoint or clearly mark the API as an experimental legacy subsystem outside the supported demo.

### WTA and live-provider completeness

- **Current state:** WTA and ATP live-score normalization code exists, and provider behavior is tested with fixtures/mocks.
- **What is missing:** `deuce/ingestion/wta.py` is a declared placeholder; external provider reliability was not verified live; API ingest silently returns row-level errors in a success response.
- **Why it matters:** The broader architecture is not production-ready for comprehensive ATP/WTA ingestion.
- **Recommended next action:** Decide whether this subsystem remains supported. If yes, add provider contract tests, observable error status, and WTA ranking/stat ingestion.

## 5. Broken, Risky, or Misleading Areas

| Priority | Finding | Evidence and impact |
|---|---|---|
| Critical | No critical code-execution or exposed-secret issue was found | Secret scan found configuration names but no committed credentials, private keys, or tokens. |
| High | The audited demo is not in Git history | `git status` shows 19 modified/untracked entries; `origin/main` contains only the initial commit. Clean-clone and deployment claims are currently false for the committed repository. |
| High | Missing inputs are presented as explanatory signals | Default prediction cites seed status even though `seed_diff_missing`, `seeded_flag_a_missing`, and `seeded_flag_b_missing` are all one. This can misrepresent why the model produced its result. |
| High | Walk-forward evaluation is not strictly separated by date | Fold reports show equal calibration/OOS boundary dates, including 2020-09-14, 2021-11-15, 2023-05-08, and 2024-07-15. |
| High | Public release is incomplete | README live URL is a placeholder and all four referenced screenshots are missing. |
| Medium | Dependency verification used a noncompliant global environment | Python 3.11.9 worked, but installed pandas 1.5.3 is below `requirements.txt`'s `>=2.0`, numpy is 1.23.5, and `pip check` reports multiple unrelated global conflicts. Tests did not run in a clean project venv. |
| Medium | Documentation describes conflicting product scopes | `README.md` presents only the portfolio demo, while `docs/architecture.md`, `docs/development.md`, `docs/LAYOUT.md`, and `docs/roadmap.md` describe the larger live-score/market/API product. `docs/LAYOUT.md` still says `app.py` is the markets/research UI. |
| Medium | API prediction is a placeholder | `deuce/api/routes/predictions.py` always returns `available: false`. |
| Medium | “Backtest” artifacts contain zero bankroll metrics | `walk_forward_report.json` reports ROI, yield, drawdown, edge, and bets as zero because OOS rows contain no market prices. The default UI avoids displaying those fields, but the stored report can be mistaken for a completed economic backtest. |
| Medium | Rebuild artifacts are not byte reproducible | Model pickle and sidecar hashes changed on repeated builds due to timestamps and absolute paths, although predictions and semantic metrics remained stable. |
| Medium | Dataset provenance and redistribution terms are insufficiently documented | The data is described only as “TennisMyLife-style”; no repository license, dataset license, or precise upstream citation was found. This is a public-deployment risk. |
| Medium | Direct dependency is implicit | `deuce/presentation/demo_app.py` imports Altair directly, but `altair` is not listed explicitly in `requirements.txt`; it currently arrives transitively through Streamlit. |
| Medium | Current CI has not evaluated the demo changes | CI exists, but uncommitted files cannot have passed GitHub Actions. |
| Low | Structured CLI root help falls through to legacy help | `python -m deuce.cli --help` displays the legacy Kalshi CLI because unknown first arguments delegate to `cli.py`. |
| Low | Browser chart warnings occur | Vega/Altair emitted infinite-extent and scale-binding warnings while rendering contribution/probability charts. No visible page failure was observed. |
| Low | Health-report script is stale | `scripts/project_health_report.py` misses nested `artifacts/demo/*.pkl`, still names the old project, and recommends changes that have already been superseded. |
| Low | Dead/duplicate compatibility surface remains large | Root `model_*.py`, `backtest.py`, `player_db.py`, `name_utils.py`, `tournament_catalog.py`, `evaluation.py`, the 72 KB legacy CLI, and `legacy_app.py` coexist with canonical package modules. Some are intentional shims, but the supported boundary is unclear. |
| Low | Error handling is broad in several provider and presentation paths | Multiple `except Exception: pass/continue/return []` paths make unavailable external providers difficult to diagnose. This does not affect the default offline demo. |

## 6. Current User-Facing Experience

From the default entry point, `streamlit run app.py`, a user can currently:

### Fully functional capabilities

- Open an Overview page showing the 1,232-match dataset and verified OOS metrics.
- Open Match Predictor and select two players from the local player database/demo intersection.
- Select hard, clay, or grass context.
- Receive model-backed win probabilities, a confidence label, data-completeness score, deterministic explanation, factor chart, and grouped contributions.
- Open Backtest Explorer and view accuracy, ROC-AUC, Brier score, log loss, calibration, confidence buckets, probability distribution, and fold-level evaluation data.
- Open Methodology and review the local workflow, limitations, and validation command.

### Demo-only capabilities

- Player rankings and points are a static local snapshot last updated on March 15, 2026.
- Predictions use no injuries, withdrawals, travel, live form, current event draw, current seed, or live match state.
- The default model is a compact educational Core model and should not be presented as a production forecasting service.
- “Backtest Explorer” is a model-validation explorer, not a verified financial/market backtest.

### Unavailable or incomplete capabilities

- No public Streamlit deployment is linked.
- README screenshots are unavailable.
- No real HTTP prediction endpoint exists.
- WTA ranking/stat ingestion is not implemented.
- Legacy live-market and provider workflows are hidden from the default app and were not verified against current external services.
- The default explanation can currently surface unavailable input categories as if they were observed.

The hidden `legacy_app.py` still exposes Trading desk, Match lab, Research & data, and Developer tabs, including live providers, market data, ATP context, model operations, and bankroll/edge presentation. It is not the default entry point and requires additional artifacts/providers for full operation.

## 7. Verification Results

Commands executed:

| Command / check | Result |
|---|---|
| `python --version` | Pass: Python 3.11.9 |
| Dependency version imports | Pass for runtime imports; Streamlit 1.56.0, pandas 1.5.3, numpy 1.23.5, scikit-learn 1.5.2 |
| `python -m pip check` | Fail: global environment contains multiple dependency conflicts |
| `python -m deuce.cli demo validate` | Pass: dataset 1,232; OOS 1,072; accuracy 0.6129; ROC-AUC 0.6255; Brier 0.2397 |
| `python -m py_compile ...` for app/demo/legacy entry modules | Pass |
| `python -m pytest tests -q` | Pass: 133 passed, 4 warnings in 10.16 seconds |
| `python -m deuce.cli feature-tiers` | Pass |
| `python -m streamlit run app.py --help` | Pass |
| Actual Streamlit server on `127.0.0.1:8501` | Pass |
| Browser navigation across all four default pages | Pass |
| Default Alcaraz vs. Sinner prediction | Pass: 53.1% vs. 46.9%, Low confidence, approximately 82% completeness |
| Browser console inspection | No errors; Vega/Altair warnings observed |
| `python -m deuce.cli demo build` repeated | Pass both times; metrics and OOS output stable |
| Repeated-build hash comparison | OOS CSV/report and semantic metrics identical; model pickle and feature sidecar not byte-identical |
| FastAPI TestClient `/health` | Pass: HTTP 200 |
| FastAPI TestClient `/predictions/live/demo` | HTTP 200 but intentionally unavailable placeholder |
| FastAPI mock-provider ingest | Pass through mocked provider; covered more reliably by automated tests |
| `git diff --check` | Pass |
| Secret-pattern scan | No exposed secrets found |
| Git history/status inspection | One commit only; current demo release is uncommitted |

Important warnings:

- `requests` reported an installed dependency-version warning.
- FastAPI tests reported three `httpx` deprecation warnings.
- The global Python installation does not satisfy the documented dependency set, even though the project tests pass in it.

Environment assumptions required to run:

- Python 3.11.
- A dedicated virtual environment with `requirements.txt` installed.
- Repository root as the working directory.
- For the default demo: `data/demo/`, `data/player_stats.json`, `data/tournament_catalog.json`, and all six `artifacts/demo/` runtime files.
- No API keys or external providers are required for the default app.
- External provider, legacy market, and Playwright workflows have additional network/browser assumptions and were not required for the default demo.

## 8. Recommended Next Milestone

### Milestone name

**Release-Candidate Model Trust and Packaging**

### Objective

Make the existing portfolio demo defensible, reproducible, and deployable without expanding the product scope.

### Concrete deliverables

1. Suppress contribution/explanation factors when their source inputs are missing.
2. Group temporal splits by date and enforce strictly earlier calibration data.
3. Rebuild the demo artifacts and update all reported metrics.
4. Strengthen artifact validation with checksums, feature-schema checks, and expected package/model metadata.
5. Remove machine-specific absolute paths from committed artifact metadata.
6. Align README and supporting docs with the supported default product.
7. Add real screenshots, commit the complete release set, and verify CI from a clean checkout.

### Acceptance criteria

- Default Alcaraz/Sinner explanation does not mention seed, entry, home, demographic, or event factors unless those inputs are present.
- Every fold satisfies `max(train) <= max(calibration) < min(OOS)` with no date appearing across calibration and OOS.
- Demo build and validation tests pass in a clean Python 3.11 virtual environment.
- Repeated builds produce identical semantic model metadata and prediction outputs; volatile timestamps are explicitly excluded from reproducibility claims.
- `python -m pytest tests -q` and CI pass.
- `git status` is clean after the release commit.
- README has a valid live URL and all referenced screenshots render.

### Files likely affected

- `deuce/presentation/demo_helpers.py`
- `deuce/presentation/demo_app.py`
- `deuce/training/walk_forward_folds.py`
- `deuce/training/temporal.py`
- `deuce/training/train.py`
- `deuce/demo/build.py`
- `deuce/demo/artifacts.py`
- `tests/test_demo_helpers.py`
- `tests/test_demo_pipeline.py`
- `tests/test_walk_forward.py`
- `README.md`
- `docs/LAYOUT.md`
- `docs/development.md`
- `artifacts/demo/*`
- `docs/images/*`

### Risks and dependencies

- Strict date-grouped folds will change the reported metrics.
- Explanation filtering must remain consistent with scaled model contributions.
- Pickled scikit-learn artifacts remain version-sensitive.
- Public release depends on confirming dataset provenance and redistribution permission.
- Deployment requires repository access and a chosen Streamlit Community Cloud app.

## 9. Prioritized Roadmap

| Priority | Milestone | Expected impact | Complexity | Dependency |
|---|---|---|---|---|
| 1 | Release-Candidate Model Trust and Packaging | Makes current claims defensible and creates a shippable demo | Medium | None |
| 2 | Clean-Environment CI and Deployment | Proves clean-clone behavior and provides a live portfolio URL | Small | Milestone 1 |
| 3 | Documentation and Legacy-Scope Consolidation | Removes contradictory product framing and clarifies supported vs. archived components | Medium | Milestone 1 |
| 4 | Real Prediction API or Explicit API De-scope | Resolves the misleading placeholder HTTP prediction surface | Medium | Product-scope decision |
| 5 | Data/Model Quality Expansion | Adds stronger features, date-aware current data, segmented evaluation, and drift monitoring | Large | Stable release and provenance decision |

## 10. Questions / Decisions Needed

1. Is the supported product now exclusively the portfolio demo, or must the legacy live-score/market/API stack remain a maintained first-class product?
2. What is the authoritative source and redistribution license for the historical match files and player-stat snapshot?
3. Should the public demo remain ATP-focused, or is WTA support an actual release requirement?
4. Which GitHub repository/branch and Streamlit Community Cloud account should own the deployment?
