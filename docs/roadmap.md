# Roadmap

## Done (reconstruction phases)

- **Domain** — `ScoringEngine`, Pydantic models, validation tests.
- **Providers** — ABC, legacy sync bridge, mock/replay, health, normalization to `MatchState`.
- **Storage** — SQLite + SQLAlchemy ORM + `MatchRepository`.
- **Pipeline** — Events, dedup, reconciliation façade, async retry, poller.
- **Services** — `ProviderManager`, `LivePredictionService` (lazy ML import), `MatchOrchestrator`.
- **API** — FastAPI routes and Uvicorn entrypoint.

## Next

1. **Wire predictions** — API/CLI to accept `PlayerStats` or load from `player_db` for real `predict_match_probability` calls.
2. **Auth & rate limits** — If the API is exposed beyond localhost.
3. **Postgres** — Swap `DEUCE_DB_URL` for production; keep SQLite for local dev.
4. **WebSocket** — Push score updates instead of poll-only clients.
5. **Promote scripts** — Move recurring data jobs into `deuce.cli` subcommands.
6. **CI** — `pip install -r requirements.txt` + `pytest` on each push.
