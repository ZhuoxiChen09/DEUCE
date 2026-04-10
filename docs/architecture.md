# System architecture

DEUCE layers build from domain primitives upward: scoring state, live providers, optional persistence, and an HTTP API.

## Layers

```
FastAPI (deuce.api)
    → Services (MatchOrchestrator, ProviderManager, LivePredictionService)
        → Pipeline (poller, dedup, reconciliation, retry)
            → Providers (LiveScoreProvider + live_scores backends)
            → Storage (SQLite + MatchRepository)
Domain (MatchState, ScoringEngine, validation)
```

## Data flow (live ingestion)

1. **Provider** returns normalized dict rows (`deuce/live_scores.py` shape).
2. **Reconciliation** maps rows to `MatchState` via `deuce.providers.normalization`.
3. **Deduplicator** drops unchanged score fingerprints per `(match_key, source)`.
4. **Repository** upserts `MatchRow`, appends `EventRow`, stores `SnapshotRow`.
5. **API** exposes health, matches, ingest, simulation reconcile, and provider health metrics.

## Key packages

| Package | Role |
|---------|------|
| `deuce.domain` | Enums, Pydantic models, `ScoringEngine`, validation |
| `deuce.providers` | `LiveScoreProvider` ABC, mock/replay, health, normalization |
| `deuce.storage` | SQLAlchemy models, SQLite engine, `MatchRepository` |
| `deuce.pipeline` | Events, dedup, `ScoreReconciler`, async retry, `LiveScorePoller` |
| `deuce.services` | `ProviderManager`, `LivePredictionService`, `MatchOrchestrator` |
| `deuce.api` | FastAPI app, routes, Uvicorn entrypoint |

## Environment

- `DEUCE_DB_URL` — SQLAlchemy URL (default: `sqlite:///<DEUCE_DATA_DIR>/deuce.db`).
- `DEUCE_DATA_DIR` — Directory for default SQLite file (default `.deuce`).
- `DEUCE_LIVE_SCORE_*` — See `docs/provider_strategy.md` and `deuce/live_scores.py`.
