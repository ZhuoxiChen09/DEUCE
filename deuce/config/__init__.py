"""Central configuration and pipeline documentation."""

from deuce.config.settings import Tour, default_artifacts_dir, project_name

__all__ = ["Tour", "default_artifacts_dir", "project_name", "PIPELINE_STEPS"]

# Canonical quant pipeline (documented for README and onboarding).
PIPELINE_STEPS = (
    "market_odds + player/tournament context",
    "player normalization / join (DB + rankings + optional leaderboards)",
    "pre-match feature engineering (tier: core | enhanced | legacy-research)",
    "model probability (logistic + optional Platt calibration)",
    "calibration (bundle holds calibrated head)",
    "no-vig or raw implied market probability",
    "edge = model - fair_market",
    "bet filter / no-bet (threshold, liquidity, completeness)",
    "stake sizing (flat | fractional Kelly)",
    "backtest logging & performance analytics",
)
