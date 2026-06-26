# DEUCE

DEUCE is an offline Streamlit portfolio demo for transparent tennis match prediction and strict walk-forward model evaluation.

The supported product is the four-page app launched from `app.py`: Overview, Match Predictor, Backtest Explorer, and Methodology. It uses committed historical data and a committed calibrated logistic-regression bundle. No API keys, live providers, odds, or external services are required.

No public deployment is currently claimed or linked.

## Private release gate

This release candidate is safe to commit and push only to a private remote. The
bundled demo dataset and derived artifacts are not cleared for public
redistribution until upstream license or permission scope is confirmed. Do not
make this repository public or deploy the bundled demo publicly before that
clearance exists. See `THIRD_PARTY_NOTICES.md`.

## Release-candidate results

The committed demo dataset contains 1,232 historical matches. Strict date-grouped walk-forward evaluation produces 1,072 out-of-sample predictions:

- Accuracy: 61.57%
- ROC-AUC: 0.6191
- Brier score: 0.2419
- Log loss: 0.6779

Every fold uses whole calendar-date groups and satisfies:

```text
max(train_date) <= max(calibration_date) < min(oos_date)
```

These metrics describe the included dataset and evaluation procedure. They are not claims about future performance.

## Release and reproduction

Python 3.11 is required, matching `runtime.txt`.

```bash
git clone <repository-url>
cd DEUCE
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install dependencies and run the complete verification sequence:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests -q
python -m deuce.cli demo build
python -m deuce.cli demo validate
python -m deuce.cli feature-tiers
python -m py_compile app.py deuce/presentation/demo_app.py deuce/presentation/demo_helpers.py
git diff --check
```

Launch Streamlit:

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

The artifact build reads `data/demo/demo_matches.csv` and writes the release bundle to `artifacts/demo/`. Semantic outputs, schemas, metrics, and checksums are deterministic. Timestamp and host details are isolated in `artifacts/demo/build_metadata.json`, which is explicitly marked volatile.

## Model and explanation integrity

The demo uses leakage-aware pre-match features, standardized logistic regression, and Platt calibration. Explanations use deterministic coefficient contributions.

Unavailable seed, entry, event, demographic, home-context, or match-progression inputs are not presented as observed predictive factors. Their explicit missingness indicators may still affect the model and are labeled separately.

## Supported and retained scope

The default Streamlit portfolio demo is the supported release surface.

The following repository areas are retained for research history or future work but are out of scope for this portfolio release:

- `legacy_app.py`: archived legacy Streamlit market/research interface
- `deuce/api/`: experimental FastAPI foundation; its prediction endpoint is not a supported product feature
- `deuce/live_scores.py` and `deuce/providers/`: experimental external-provider integrations
- `deuce/market/`, `deuce/edge/`, and `deuce/bankroll/`: legacy market-analysis modules
- root compatibility shims such as `model_train.py` and `backtest.py`

These components are not required to build, validate, or run the default demo.

## Project structure

```text
DEUCE/
|-- app.py
|-- deuce/
|   |-- demo/                 # Artifact build and validation
|   |-- presentation/         # Four-page Streamlit demo
|   |-- training/             # Features, training, calibration, prediction
|   `-- research/             # Walk-forward evaluation
|-- data/demo/                # Deterministic committed dataset
|-- artifacts/demo/           # Lightweight committed runtime bundle
|-- docs/
|   |-- demo/                 # Demo script and walkthrough notes
|   |-- release/              # Release checklist and clean-clone verification
|   `-- project/              # Project status and audit records
`-- tests/
```

## Limitations

- The demo is a compact ATP-focused historical slice.
- Historical fields can be incomplete or noisy.
- The model does not know injuries, withdrawals, travel disruption, tactical changes, or live match conditions.
- Player context is a static local snapshot, not a live ranking service.
- The model is intentionally simple and dataset-specific.
- Dataset provenance and redistribution terms still require explicit release-owner confirmation.
- Public deployment and public repository distribution are blocked until demo
  data and artifact redistribution rights are cleared.

This is an educational sports-analytics demo, not betting advice or a guarantee of future outcomes.
