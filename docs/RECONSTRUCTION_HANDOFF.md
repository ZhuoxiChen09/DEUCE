# DEUCE Reconstruction Handoff

**Date**: 2026-04-05 (updated)  
**Canonical tree**: repository root (`deuce/*`, `tests/*`, `docs/*`). A copy may still exist under `.claude/worktrees/` for git worktree workflows — use **one** checkout for daily work.

---

## Status: Phases 1–7 implemented

### Phase 1: Domain Foundation

`deuce/domain/` — enums, Pydantic models, `ScoringEngine`, validation. Tests: `tests/test_domain_scoring.py`.

### Phase 2: Provider Abstraction

`deuce/providers/` — `LiveScoreProvider`, sync bridge to `live_scores`, normalization, health, mock, replay. Tests: `tests/test_providers.py`.

### Phase 3: Storage

`deuce/storage/` — SQLAlchemy ORM (`MatchRow`, `EventRow`, `SnapshotRow`, `PredictionRow`, `RawPayloadRow`), SQLite engine (`StaticPool` + `check_same_thread=False` for in-memory + threaded TestClient), `MatchRepository`. Tests: `tests/test_storage_repository.py`.

### Phase 4: Pipeline

`deuce/pipeline/` — `MatchEvent` / `EventType`, `apply_event`, `EventDeduplicator`, `ScoreReconciler`, `retry_async`, `LiveScorePoller`. Tests: `tests/test_pipeline_events.py`.

### Phase 5: Services

`deuce/services/` — `ProviderManager`, `LivePredictionService` (lazy ML import), `MatchOrchestrator`. Tests: `tests/test_orchestrator.py`.

### Phase 6: API

`deuce/api/` — FastAPI `create_app`, routes (`health`, `matches`, `ingest`, `predictions`, `providers`, `simulation`), `python -m deuce.api.server`. Tests: `tests/test_api_deuce.py`.

### Phase 7: Docs & requirements

- `docs/architecture.md`, `docs/provider_strategy.md`, `docs/development.md`, `docs/roadmap.md`
- `requirements.txt`: `sqlalchemy`, `fastapi`, `uvicorn`, `httpx<0.28` (TestClient), `pytest-asyncio`, `numpy<2`
- `pytest.ini`: `asyncio_mode = auto`
- `README.md` + `scripts/dev.ps1` include HTTP API

---

## Verify

```powershell
cd .claude\worktrees\hardcore-chatterjee
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests\ -q
```

Use a **clean venv**; global installs with `numpy 2.4+` and old `sklearn` wheels can fail collection.

---

## Key references

| Topic | Location |
|-------|----------|
| Scoring | `deuce/domain/scoring.py` |
| Provider rows → domain | `deuce/providers/normalization.py` |
| Legacy fetch | `deuce/live_scores.py` |
| API entry | `deuce/api/server.py` |

---

## Gotchas (unchanged)

- Pydantic v2: `model_copy`, `ConfigDict(frozen=True)`.
- `ServerSide` vs player names — map in normalization.
- Tiebreak server rotation — see domain tests.
- `httpx` 0.28+ breaks Starlette `TestClient` — pin `<0.28`.
