# Scripts

These scripts are not all equal. Some are part of normal project workflows; others are one-off research or audit helpers.

## Core Workflow Scripts

- `build_historical_from_db.py`: build a training-style historical CSV from a match list plus player DB
- `build_historical_from_tml.py`: build a training CSV from TennisMyLife datasets
- `fetch_atp_activity.py`: fetch ATP player activity into a match-list CSV
- `fetch_espn_matches.py`: fetch ESPN results into a match-list CSV
- `run_espn_training_pipeline.py`: convenience pipeline for ESPN fetch -> dataset build -> training

## Audit And Reporting Scripts

- `dataset_alignment_audit.py`: inspect historical dataset alignment
- `model_diagnostic_report.py`: generate model diagnostics
- `player_order_bias_audit.py`: check A/B order bias in the prediction pipeline
- `post_fix_quality_audit.py`: broader quality audit after training fixes

The obsolete `project_health_report.py` was removed: it inspected the old flat
artifact layout and reported outdated project recommendations. Use
`python -m deuce.cli demo validate` for the supported bundle and the
[research roadmap](../docs/roadmap.md) for current priorities.

## One-Off Exploration

- `find_leaderboard_api.py`: exploratory script for locating ATP leaderboard data sources

## Dev shortcuts (`dev.ps1`)

From repo root: `powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 setup|test|app|api|tiers`

- `api` runs `python -m deuce.api.server` (FastAPI; open `/docs`).

## Guidance

- Prefer the app, `python -m deuce.cli`, and the library modules for regular use.
- Use the scripts in this folder when you are doing data collection, audits, or one-off research support work.
- If a script becomes part of the normal product flow, promote it into the `deuce/` package or a documented CLI command.
