# DEUCE Demo Script (Draft)

Target length: 60-90 seconds.

## 1. Landing Context

"DEUCE is a transparent tennis match prediction and backtesting lab. The demo
uses a small committed historical dataset, so every number shown here can be
reproduced locally."

Show:

- One-sentence product description
- Data and artifact status
- Build command: `python -m deuce.cli demo build`

## 2. Match Predictor

"I can select two players from the curated player dataset and compare them
through the same Core model used in the training pipeline."

Show:

- Two player selectors
- Win probability for each player
- Predicted winner
- Confidence / data completeness
- Top input signals such as ranking difference, points difference, surface
  context, and seed/context fields only when those source inputs are available

## 3. Backtest

"The model is evaluated out of sample using date-grouped walk-forward folds, so
every evaluation date is strictly later than the calibration window."

Show:

- Number of OOS predictions
- Accuracy
- ROC-AUC
- Brier score
- Confidence-threshold table
- Calibration chart or prediction-vs-outcome bins

## 4. Methodology

"The methodology page explains what data is used, how features are built, how
validation avoids future information, and what the model cannot know."

Show:

- Data source and filtering rules
- Core feature set and leakage guard
- Walk-forward validation design
- Known limitations: compact dataset, ATP-only slice, historical context gaps,
  no guarantee of future performance

## 5. Close

"The important engineering point is reproducibility: clone the repo, install
requirements, run one build command, and the demo regenerates the model,
walk-forward report, metrics, and manifest from committed data."
