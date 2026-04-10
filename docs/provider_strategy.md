# Live score provider strategy

## Normalized row shape

All backends in `deuce/live_scores.py` converge to dicts with (among others):

- `date`, `tournament`, `draw`, `status_text`
- `is_live`, `is_completed`, `is_scheduled`
- `players`: `[{ "name", "score_tokens", "country_code"? }, ...]` (two singles players)
- `source`, `serve` (where available)

## `DEUCE_LIVE_SCORE_PROVIDER`

Handled by `load_live_score_provider_config()` and `fetch_scoreboard_by_dates()`:

| Value | Behavior |
|-------|----------|
| `auto` | Official feed (if configured) → WTA + ATP public → Miami Open public → ESPN |
| `official` | Configured official JSON only |
| `officialpublic` | WTA + ATP public, else ESPN |
| `wtapublic` / `atppublic` / `miamiopen` / `espn` | Force that chain or source |

## Pluggable async layer

`deuce.providers.default_live_score_provider()` wraps `fetch_scoreboard_by_dates` as a `LiveScoreProvider`.

`ProviderManager` (failover mode) tries a list of providers in order until one returns without raising.

## Health metrics

`InstrumentedLiveScoreProvider` records success/failure and latency on `ProviderHealthTracker`. The API exposes `GET /providers/health` after at least one instrumented fetch.

## Domain mapping

`infer_server_side()` maps `serve` plus `source` (ATP 0/1 vs WTA 1/2) to `ServerSide`. `match_state_from_provider_dict()` builds `MatchState` for storage and simulation.
