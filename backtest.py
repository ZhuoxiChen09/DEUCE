"""
Backtesting: probability metrics (legacy API) and full bankroll / ROI research backtest.

Prefer importing from ``deuce.backtest`` for new code.
"""

from typing import Any, Dict, Sequence

from deuce.backtest.engine import BacktestConfig, BacktestResult, run_backtest, summarize_backtest, summarize_legacy_predictions


def evaluate_predictions(
    predictions: Sequence[Dict[str, Any]],
    outcomes: Sequence[int],
) -> Dict[str, float]:
    """
    Log loss and Brier score vs outcomes (0 = B won, 1 = A won).

    ``roi`` and ``sharpe_ratio`` remain 0 unless you run ``run_backtest`` with prices.
    """
    return summarize_legacy_predictions(predictions, outcomes)


__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "evaluate_predictions",
    "run_backtest",
    "summarize_backtest",
    "summarize_legacy_predictions",
]
