# DEUCE

DEUCE is a Python toolkit for pre-match ATP tennis prediction. The offline portfolio demo is a four-page Streamlit app. The product service is a FastAPI application that serves the same committed, calibrated logistic-regression model.

## Research direction

**Do court conditions change the predictive value of a player's serve and return strengths beyond ranking and surface type?**

The current demo is a baseline. Court-speed integration and the experiments below
are planned work, not capabilities or improvements already demonstrated.

- [Research questions and experiment protocol](docs/research/RESEARCH_PLAN.md)
- [Prioritized roadmap](docs/roadmap.md) and [GitHub milestones](https://github.com/ZhuoxiChen09/DEUCE/milestones)
- [Research issues](https://github.com/ZhuoxiChen09/DEUCE/issues?q=is%3Aissue%20is%3Aopen%20label%3Aresearch)
- [Documentation guide](docs/README.md)

Start with [evaluation parity](https://github.com/ZhuoxiChen09/DEUCE/issues/1),
[CI reproducibility](https://github.com/ZhuoxiChen09/DEUCE/issues/2), and
[ranking/Elo baselines](https://github.com/ZhuoxiChen09/DEUCE/issues/3).
The [distribution-scope issue](https://github.com/ZhuoxiChen09/DEUCE/issues/9)
tracks the mismatch between current public visibility and the release notice below.

## Current demo

The offline demo is the four-page app launched from `app.py`: Overview, Match Predictor, Backtest Explorer, and Methodology. It uses committed historical data and a committed calibrated logistic-regression bundle. No API keys, live providers, odds, or external services are required. The FastAPI service, described below, is the product path and is not required to run the demo.

No public deployment is currently claimed or linked.

## Private release gate

This release candidate is safe to commit and push only to a private remote. The
bundled demo dataset and derived artifacts are not cleared for public
redistribution until upstream license or permission scope is confirmed. Do not
make this repository public or deploy the bundled demo publicly before that
clearance exists. See `THIRD_PARTY_NOTICES.md`.

## Release-candidate results

The committed demo dataset contains 1,232 historical matches. Strict date-grouped walk-forward evaluation produces 1,072 out-of-sample predictions:

- Accuracy: 61.57%
- ROC-AUC: 0.6191
- Brier score: 0.2419
- Log loss: 0.6779

Every fold uses whole calendar-date groups and satisfies:

```text
max(train_date) <= max(calibration_date) < min(oos_date)
```

These metrics describe the included dataset and evaluation procedure. They are not claims about future performance.

## Release and reproduction

Python 3.11 is required, matching `runtime.txt`.

```bash
git clone <repository-url>
cd DEUCE
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install dependencies and run the complete verification sequence:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m deuce.cli demo build
python -m deuce.cli demo validate
python -m deuce.cli feature-tiers
python -m py_compile app.py deuce/presentation/demo_app.py deuce/presentation/demo_helpers.py
git diff --check
```

Launch Streamlit:

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

The artifact build reads `data/demo/demo_matches.csv` and writes the release bundle to `artifacts/demo/`. Semantic outputs, schemas, metrics, and checksums are deterministic. Timestamp and host details are isolated in `artifacts/demo/build_metadata.json`, which is explicitly marked volatile.

## Product API

FastAPI serves pre-match win probabilities from `artifacts/demo/tennis_model_core.pkl`. Player strengths come from `data/player_stats.json` for named matchups, or from the pre-match fields of an imported historical row. Live scores are not model inputs. Scrapers stay off unless `DEUCE_LIVE_SCORES_ENABLED=1`.

Secrets and deployment settings are environment variables. Copy `.env.example` to `.env` for local overrides and do not commit `.env`.

### API on SQLite

SQLite is the local default when `DEUCE_DB_URL` is unset. Tables are created on startup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m deuce.api.server
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

`GET /predictions/live/demo` scores Carlos Alcaraz vs Jannik Sinner on hard courts. Query parameters `player_a`, `player_b`, and `surface` (`Hard`, `Clay`, `Grass`, or `Carpet`) score another matchup.

### Docker Compose

Postgres is the product database. The API container runs `alembic upgrade head` before it listens.

```bash
docker compose up --build
```

The API is published on port 8000 and Postgres on port 5432. Override `POSTGRES_PASSWORD` and the other values in `.env.example` for anything other than a local bootstrap.

### Data import

The default adapter is `local_csv`. It reads `DEUCE_IMPORT_SOURCE` (default `data/demo/demo_matches.csv`), which is already in the repository. A new dataset is a new `DataSourceAdapter` registered with `register_adapter`. This tree does not add another public dataset.

```bash
python -m deuce.jobs.import_data
```

The same command is `python -m deuce.cli import-data`. After Compose is up:

```bash
docker compose exec api python -m deuce.jobs.import_data
```

`POST /imports` runs one import inside the API process. `GET /predictions/historical/{match_key}` scores a stored row.

Data-quality checks cover schema, required nulls, duplicate match keys, date range, value ranges, and referential sanity (the two players are distinct, and the winner side resolves to one of them). `DEUCE_DQ_MODE=block` is the default: any error-level failure writes no match rows, logs the issues, and stores an `import_runs` row with status `blocked`. `DEUCE_DQ_MODE=flag` loads the valid rows, skips the bad ones, and logs a `flagged` run. A missing required column is always blocked. Warning-only issues, such as an unknown tournament level, flag the run and still load the row. Exit code 0 means rows were loaded (`loaded` or `flagged`). Exit code 1 means the load was blocked or the source could not be read.

Set `DEUCE_IMPORT_ENABLED=1` to schedule that job inside the API. The default interval is 1440 minutes. `DEUCE_IMPORT_ON_START=1` runs one import when the process starts.

## Model and explanation integrity

The demo uses leakage-aware pre-match features, standardized logistic regression, and Platt calibration. Explanations use deterministic coefficient contributions.

Unavailable seed, entry, event, demographic, home-context, or match-progression inputs are not presented as observed predictive factors. Their explicit missingness indicators may still affect the model and are labeled separately.

## Supported and retained scope

The Streamlit portfolio demo remains the offline release surface. The FastAPI service is the product path for predictions, Postgres, and scheduled import.

The following repository areas are retained for research history or future work and are not required to build or run the offline demo:

- `legacy_app.py`: archived legacy Streamlit market/research interface
- `deuce/live_scores.py` and `deuce/providers/`: live-score integrations, off by default on the product path
- `deuce/market/`, `deuce/edge/`, and `deuce/bankroll/`: legacy market-analysis modules
- root compatibility shims such as `model_train.py` and `backtest.py`

These components are not required to build, validate, or run the default demo.

## Project structure

```text
DEUCE/
|-- app.py                    # Offline Streamlit demo
|-- Dockerfile
|-- docker-compose.yml        # API + Postgres
|-- alembic/                  # Postgres schema migrations
|-- deuce/
|   |-- api/                  # Product FastAPI service
|   |-- demo/                 # Artifact build and validation
|   |-- ingestion/            # Data-source adapters, DQ checks, import job
|   |-- jobs/                 # Scheduled import
|   |-- presentation/         # Four-page Streamlit demo
|   |-- training/             # Features, training, calibration, prediction
|   `-- research/             # Walk-forward evaluation
|-- data/demo/                # Deterministic committed dataset
|-- artifacts/demo/           # Lightweight committed runtime bundle
|-- docs/
|   |-- demo/                 # Demo script and walkthrough notes
|   |-- release/              # Release checklist and clean-clone verification
|   `-- project/              # Project status and audit records
`-- tests/
```

## Limitations

- The demo is a compact ATP-focused historical slice.
- Historical fields can be incomplete or noisy.
- The model does not know injuries, withdrawals, travel disruption, tactical changes, or live match conditions.
- Player context is a static local snapshot, not a live ranking service.
- The model is intentionally simple and dataset-specific.
- Dataset provenance and redistribution terms still require explicit release-owner confirmation.
- Public deployment and public repository distribution are blocked until demo
  data and artifact redistribution rights are cleared.

This is an educational sports-analytics demo, not betting advice or a guarantee of future outcomes.
