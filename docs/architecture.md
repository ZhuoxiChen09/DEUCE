# System architecture

## Supported portfolio-demo path

```text
data/demo/demo_matches.csv
    -> deuce.training.features
    -> calibrated logistic regression
    -> date-grouped walk-forward evaluation
    -> artifacts/demo
    -> deuce.presentation.demo_app
    -> app.py
```

The default app reads committed local files and does not call external providers.
Artifact validation checks expected files, ordered feature-schema identity, model
metadata, row counts, checksums, metric consistency, and strict fold date windows.

## Product service path

```text
data/demo/demo_matches.csv
    -> local_csv adapter
    -> data-quality checks
    -> Postgres (Alembic) historical_matches
    -> FastAPI /predictions
    -> artifacts/demo/tennis_model_core.pkl
```

Named matchups join `data/player_stats.json` and call the same pre-match feature
pipeline the Streamlit demo uses. Live-score providers are not on this path
unless `DEUCE_LIVE_SCORES_ENABLED=1`.

## Retained experimental and archived subsystems

The repository also retains earlier platform work:

```text
providers -> reconciliation -> SQLite -> services -> FastAPI
```

- `deuce.api`: product HTTP service; predictions use the committed calibrated model
- `deuce.live_scores` and `deuce.providers`: optional live-score integrations, off by default
- `deuce.market`, `deuce.edge`, and `deuce.bankroll`: legacy analysis modules
- `legacy_app.py`: archived legacy Streamlit interface

These are not part of the supported release-candidate demo and are not required
for its setup, artifact build, validation, or runtime.
