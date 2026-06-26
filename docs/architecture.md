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

## Retained experimental and archived subsystems

The repository also retains earlier platform work:

```text
providers -> reconciliation -> SQLite -> services -> FastAPI
```

- `deuce.api`: experimental HTTP and persistence foundation
- `deuce.live_scores` and `deuce.providers`: experimental provider integrations
- `deuce.market`, `deuce.edge`, and `deuce.bankroll`: legacy analysis modules
- `legacy_app.py`: archived legacy Streamlit interface

These are not part of the supported release-candidate demo and are not required
for its setup, artifact build, validation, or runtime.
