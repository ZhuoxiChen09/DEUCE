# Repository layout

Run supported release commands from the repository root with Python 3.11.

## Intended root

The repository root is kept for project entry points, dependency/runtime files,
tool configuration, and top-level source/data/test directories.

```text
DEUCE/
|-- .github/                 # CI workflow configuration
|-- .streamlit/              # Streamlit runtime configuration
|-- .vscode/                 # Workspace Explorer visibility settings
|-- app.py                   # Supported Streamlit portfolio-demo entrypoint
|-- README.md
|-- requirements.txt
|-- runtime.txt
|-- pytest.ini
|-- artifacts/demo/          # Committed lightweight runtime bundle
|-- data/demo/               # Committed deterministic demo dataset
|-- deuce/                   # Canonical package code
|-- docs/                    # Architecture, development, release, and audit docs
|-- scripts/                 # Local maintenance and research scripts
`-- tests/                   # Automated verification
```

Operational AI-tool folders such as `.agents/`, `.claude/`, and `.codex/`
remain at the root because local tooling may depend on them. They are hidden
from the default VS Code Explorer through `.vscode/settings.json`, not moved.

## Supported release surface

| Path | Role |
|---|---|
| `app.py` | Four-page Streamlit portfolio-demo entrypoint |
| `deuce/presentation/demo_app.py` | Overview, Match Predictor, Backtest Explorer, Methodology |
| `deuce/presentation/demo_helpers.py` | Deterministic display and explanation helpers |
| `deuce/demo/` | Dataset build, artifact build, discovery, and validation |
| `deuce/training/` | Feature engineering, calibration, training, prediction, temporal folds |
| `deuce/research/` | Walk-forward evaluation and reports |
| `data/demo/` | Committed deterministic demo dataset |
| `artifacts/demo/` | Committed lightweight validated runtime bundle |
| `tests/` | Automated verification |

The supported data flow is:

```text
data/demo -> deuce.training -> deuce.research -> artifacts/demo
    -> deuce.presentation.demo_app -> app.py
```

## Documentation locations

| Path | Role |
|---|---|
| `README.md` | Root project overview, release metrics, and quickstart |
| `docs/architecture.md` | Supported architecture summary |
| `docs/development.md` | Local development commands |
| `docs/demo/DEMO_SCRIPT.md` | Demo walkthrough script |
| `docs/release/RELEASE_CANDIDATE_CHECKLIST.md` | Release-candidate checklist |
| `docs/release/CLEAN_CLONE_VERIFICATION.md` | Clean-clone validation record |
| `docs/README.md` | Documentation guide |
| `docs/research/RESEARCH_PLAN.md` | Active research questions and experiment protocol |
| `docs/roadmap.md` | Priorities and GitHub milestones |
| `docs/project/REPOSITORY_ORGANIZATION.md` | Organization and cleanup record |
| `docs/archive/PROJECT_STATUS_AUDIT.md` | Historical project status and risk audit |

## Retained root Python files

These files are intentionally retained at the root to preserve current command,
test, script, pickle, and legacy import compatibility. New package code should
prefer imports from `deuce/` when an equivalent package module exists.

| Path | Status |
|---|---|
| `app.py` | Required executable entry point for `streamlit run app.py` |
| `api.py` | Legacy Kalshi/public market helper used by tests and legacy modules |
| `atp_activity.py` | Legacy ATP activity fetcher used by `scripts/fetch_atp_activity.py` |
| `backtest.py` | Compatibility shim re-exporting `deuce.backtest.engine` |
| `cli.py` | Legacy CLI delegated to by `python -m deuce.cli fetch` and entrypoint tests |
| `espn_tennis.py` | Legacy ESPN ingestion helper used by scripts |
| `evaluation.py` | Legacy evaluation module used by legacy UI and prediction tools |
| `legacy_app.py` | Archived legacy Streamlit market/research interface |
| `model_config.py` | Legacy model configuration imported by training/prediction compatibility paths |
| `model_features.py` | Compatibility shim re-exporting `deuce.training.features` |
| `model_predict.py` | Compatibility shim re-exporting `deuce.training.predict` |
| `model_train.py` | Compatibility shim re-exporting `deuce.training.train` and pickle wrapper names |
| `name_utils.py` | Compatibility shim re-exporting `deuce.name_utils` |
| `player_db.py` | Compatibility shim re-exporting `deuce.player_db` |
| `tournament_catalog.py` | Compatibility shim re-exporting `deuce.tournament_catalog` |
| `weather_enrichment.py` | Legacy catalog weather enrichment helper imported by `deuce.tournament_catalog` |

`app_activity.py` and `clip.py` are not present in the current root.

## Retained legacy and experimental package surface

| Path | Status |
|---|---|
| `legacy_app.py` | Archived legacy market/research Streamlit interface |
| `deuce/api/` | Experimental FastAPI and persistence foundation |
| `deuce/live_scores.py`, `deuce/providers/` | Experimental external-provider integrations |
| `deuce/market/`, `deuce/edge/`, `deuce/bankroll/` | Legacy market-analysis modules |

These paths are not required to build, validate, or run the supported portfolio
demo, but they remain available for existing tests, scripts, and research
history.
