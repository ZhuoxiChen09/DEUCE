# Repository layout

Single **canonical checkout**: the repo root (this project). Run everything from here with the same `.venv` — do not treat `.claude/worktrees/*` as a second product; worktrees are optional for parallel git work only.

## Top level

| Path | Role |
|------|------|
| `app.py` | Streamlit UI entry (markets, research tabs) |
| `cli.py` | Legacy argparse CLI (Kalshi fetch, DB update); still invoked by `python -m deuce.cli` for some commands |
| `api.py` | Kalshi **HTTP** client (markets), not the FastAPI server |
| `model_train.py`, `model_predict.py`, `model_features.py` | Thin shims → `deuce.training.*` |
| `backtest.py`, `player_db.py`, `name_utils.py`, … | Thin shims → `deuce.*` for old import paths |
| `data/`, `artifacts/` | Datasets and trained bundles (artifacts gitignored) |
| `tests/` | `pytest` |
| `docs/` | Design notes + runbooks |
| `scripts/` | Data/audit one-offs + `dev.ps1` |

## Package `deuce/` (source of truth)

| Area | Path |
|------|------|
| **Domain** (tennis state machine) | `deuce/domain/` |
| **Live scores** (fetch + normalize) | `deuce/live_scores.py`, `deuce/providers/` |
| **HTTP API** (ingest, SQLite) | `deuce/api/` |
| **Pipeline / services** | `deuce/pipeline/`, `deuce/services/` |
| **Persistence** | `deuce/storage/` |
| **Training / predict** | `deuce/training/` |
| **Streamlit building blocks** | `deuce/presentation/` |
| **Pricing, bankroll, edge** | `deuce/market/`, `deuce/bankroll/`, `deuce/edge/` |
| **Features / research** | `deuce/features/`, `deuce/research/` |
| **Structured CLI** | `deuce/cli/` |
| **Shared utilities** | `deuce/name_utils.py`, `deuce/player_db.py`, `deuce/tournament_catalog.py`, … |

## Documentation map

| Doc | Content |
|-----|---------|
| `README.md` | Setup and how to run |
| `docs/LAYOUT.md` | This file |
| `docs/development.md` | API env vars, endpoints |
| `docs/architecture.md` | Layer diagram |
| `docs/provider_strategy.md` | Live score providers |
| `docs/roadmap.md` | Next steps |
| `docs/RECONSTRUCTION_HANDOFF.md` | Historical reconstruction notes |
