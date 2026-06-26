# DEUCE Demo Dataset

`demo_matches.csv` is a lightweight, reproducible portfolio-demo slice derived from
`data/historical_tml_2000_2025.csv`.

## Source

- Source file: `data/historical_tml_2000_2025.csv`
- Source origin in this repo: historical TennisMyLife-style match data already
  converted to the DEUCE training schema.
- Builder: `deuce.demo.dataset.build_demo_dataset`

## Distribution Status

The bundled demo dataset and derived files under `artifacts/demo/` are not
cleared for public redistribution until the upstream license or permission scope
is confirmed by the release owner.

Until that clearance exists, keep this release candidate on private remotes
only. Do not publish, deploy, sell, or redistribute the demo dataset or derived
artifacts publicly. This is a release-safety constraint, not a legal conclusion.

## Filtering Rules

Rows are selected deterministically with these rules:

- Date range: 2018-01-01 through 2025-12-22
- Surfaces: `Hard`, `Clay`, `Grass`
- Tournament levels: `G`, `M`, `500`, `250`, `F`, `A`, `O`
- Excludes Challenger-level rows (`D`) and rows without rank or ranking points
  for both players.
- Groups by `year + surface + tournament level`.
- Selects up to 12 rows per group using a stable SHA-256 ordering of match
  identity fields.

No matches or labels are manually fabricated.

## Shape

- Rows: 1,232
- Columns: 67
- Surfaces:
  - Hard: 584
  - Clay: 396
  - Grass: 252
- Years:
  - 2018: 152
  - 2019: 156
  - 2020: 120
  - 2021: 168
  - 2022: 156
  - 2023: 156
  - 2024: 168
  - 2025: 156

## Schema

The CSV keeps the existing historical training schema, including:

- Match identity: `player_a`, `player_b`, `winner`, `date`, `tournament`,
  `round`, `surface`, `tourney_level`
- Pre-match signals: `rank_a`, `rank_b`, `points_a`, `points_b`, `seed_a`,
  `seed_b`, `entry_a`, `entry_b`, `ioc_a`, `ioc_b`, `best_of`, `draw_size`
- Optional contextual/stat fields already present in the source schema

## Demo Fit

This slice is small enough for fast local training and walk-forward evaluation,
but it still spans multiple seasons, surfaces, and tournament tiers. That makes
it suitable for the Streamlit portfolio demo without depending on live scraping
or external APIs.
