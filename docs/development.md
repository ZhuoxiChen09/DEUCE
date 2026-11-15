# Development

## Virtual environment

Use a **dedicated venv** so `numpy`, `pandas`, and `scikit-learn` stay ABI-compatible:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` pins `numpy<2` for compatibility with current `pandas` / `scikit-learn` in this repo.

## Tests

```powershell
python -m pytest tests\ -q
```

Async tests use `pytest-asyncio` (`asyncio_mode = auto` in `pytest.ini`).

## Streamlit app

```powershell
python -m streamlit run app.py
```

This is the supported product path. Validate the committed runtime bundle first:

```powershell
python -m deuce.cli demo validate
```

## Product API (FastAPI)

The API is the product service. The Streamlit demo does not call it. Run instructions, Compose, and the import job are in the root README.

```powershell
python -m deuce.api.server
```

Postgres is selected with `DEUCE_DB_URL` and migrated with `alembic upgrade head`. SQLite remains the default for a local process and for tests. Live-score scrapers stay off unless `DEUCE_LIVE_SCORES_ENABLED=1`.

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Liveness, model loaded, scrapers enabled |
| GET | `/predictions/live/demo` | Calibrated logistic regression for a matchup |
| GET | `/predictions/historical/{match_key}` | Score an imported historical row |
| POST | `/imports` | Run one adapter import and DQ check |
| POST | `/ingest` | Live-score fetch; disabled unless scrapers are enabled |
| GET | `/matches?date=YYYYMMDD` | List stored live matches |
| GET | `/providers/health` | Tracker snapshots |
| POST | `/simulation/reconcile` | Reconcile posted rows without network |

## Windows shortcut

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 api
```
