# DEUCE

Python toolkit for **pre-match tennis research** on Kalshi-style binary contracts: Kalshi quotes, player context, calibrated win probabilities, edge signals, and bankroll backtests.

## How to run (always use this repo + `.venv`)

Work from **this repository root** only. Use the project virtualenv so dependencies match `requirements.txt` (`numpy<2` matches `pandas` / `scikit-learn`).

```powershell
cd <path-to>\kalshi_prediction_market
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional: Playwright-based flows — `python -m playwright install`  
Shortcut: `powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 setup`

### 1) Streamlit (main UI)

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**.

### 2) HTTP API (live ingest + SQLite)

```powershell
.\.venv\Scripts\python.exe -m deuce.api.server
```

Open **http://127.0.0.1:8000/docs**.  
Default DB: `.deuce/deuce.db` (overridable with `DEUCE_DB_URL` / `DEUCE_DATA_DIR`). See [`docs/development.md`](docs/development.md).

### 3) Structured CLI (preferred over root shims)

```powershell
.\.venv\Scripts\python.exe -m deuce.cli --help
.\.venv\Scripts\python.exe -m deuce.cli feature-tiers
.\.venv\Scripts\python.exe -m deuce.cli train --data data\historical_tml_2000_2025.csv --out artifacts\tennis_model_core.pkl --model-type core
```

Some commands still delegate to `cli.py` internally (e.g. Kalshi fetch); you can also run `python cli.py --help` for the legacy surface.

### `dev.ps1` shortcuts

| Task | Command |
|------|---------|
| Tests | `.\scripts\dev.ps1 test` |
| Streamlit | `.\scripts\dev.ps1 app` |
| API | `.\scripts\dev.ps1 api` |
| Feature tiers | `.\scripts\dev.ps1 tiers` |

## Repository structure

High-level map (details: [`docs/LAYOUT.md`](docs/LAYOUT.md)).

| Path | Purpose |
|------|---------|
| `deuce/` | **Canonical package** — training, pricing, live scores, domain, API, storage |
| `deuce/domain/` | Match state, `ScoringEngine`, validation |
| `deuce/providers/`, `deuce/live_scores.py` | Live score providers + normalization |
| `deuce/api/` | FastAPI app (`python -m deuce.api.server`) |
| `deuce/storage/` | SQLAlchemy + SQLite helpers |
| `deuce/pipeline/`, `deuce/services/` | Ingest pipeline, orchestration, provider manager |
| `deuce/presentation/` | Streamlit modules used by `app.py` |
| `deuce/training/` | Features, train, predict (root `model_*.py` re-export here) |
| `app.py` | Streamlit entrypoint |
| `api.py` | **Kalshi REST client** for market data (not the FastAPI server) |
| `cli.py` | Legacy CLI (still used for some `deuce.cli` dispatches) |
| `tests/` | Pytest suite (`pytest.ini`: asyncio auto) |
| `scripts/` | Data/audit scripts; `dev.ps1` |
| `docs/` | Architecture, providers, development, backtest notes |

Root files like `player_db.py`, `model_train.py`, `backtest.py` are **compatibility shims** so older imports keep working; new code should import from `deuce.*`.

## Live score providers

Controlled with `DEUCE_LIVE_SCORE_PROVIDER` and related env vars (`auto`, `espn`, `wtapublic`, `official`, …). Full table: [`docs/provider_strategy.md`](docs/provider_strategy.md).

Official JSON feed (optional):

```powershell
$env:DEUCE_LIVE_SCORE_PROVIDER="official"
$env:DEUCE_OFFICIAL_LIVE_SCORE_URL="https://your-provider.example.com/tennis/live"
$env:DEUCE_OFFICIAL_LIVE_SCORE_API_KEY="your-key"
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q
```

Use a **clean venv**; mismatched global `numpy` / `scikit-learn` wheels often break imports.

## Backtesting (library)

```python
from deuce.backtest import BacktestConfig, run_backtest

result = run_backtest(rows, config=BacktestConfig(edge_threshold=0.03, stake_mode="kelly"))
```

Legacy helper: `from backtest import evaluate_predictions` (root shim).

## CI

GitHub Actions: `.github/workflows/ci.yml`.

## More docs

- [`docs/development.md`](docs/development.md) — API endpoints and env vars  
- [`docs/architecture.md`](docs/architecture.md) — system layers  
- [`docs/BACKTEST_V1.md`](docs/BACKTEST_V1.md) — backtest v1 notes  
- [`docs/RECONSTRUCTION_HANDOFF.md`](docs/RECONSTRUCTION_HANDOFF.md) — reconstruction changelog  
