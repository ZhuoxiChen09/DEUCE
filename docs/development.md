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

## Experimental HTTP API (FastAPI)

The API is retained as an experimental subsystem. It is not required by the
portfolio demo, and `/predictions/live/demo` remains an unavailable placeholder.

```powershell
python -m deuce.api.server
```

Or:

```powershell
python -m uvicorn deuce.api.app:app --host 127.0.0.1 --port 8000
```

Environment:

- `DEUCE_API_HOST`, `DEUCE_API_PORT`, `DEUCE_API_RELOAD`
- `DEUCE_DB_URL` - database URL (default SQLite under `DEUCE_DATA_DIR`)

### Useful endpoints

| Method | Path | Notes |
|--------|------|--------|
| GET | `/health` | Liveness |
| POST | `/ingest` | Body `{"dates":["YYYYMMDD",...]}` - fetches via live provider and persists |
| GET | `/matches?date=YYYYMMDD` | List stored matches |
| GET | `/matches/{match_key}/state` | Stored domain JSON |
| GET | `/predictions/live/demo` | Placeholder when stats are not wired |
| GET | `/providers/health` | Tracker snapshots (after fetches) |
| POST | `/simulation/reconcile` | Body `{"rows":[...]}` - reconcile without network |

## Windows shortcut

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 api
```
