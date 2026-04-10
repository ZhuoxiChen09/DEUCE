# DEUCE Backtest V1

This document describes the production-usable v1 pre-match bankroll backtest in `deuce.backtest.engine`.

## Scope

- Pre-match binary tennis bets only.
- No target-match score features are used by this backtest policy.
- Decision flow: market math -> edge filter -> quality/completeness filter -> stake sizing -> bankroll simulation.

## Core formulas

- Edge:
  - `edge_a = model_prob_a - market_prob_a`
  - `edge_b = model_prob_b - market_prob_b`
- Pricing convention:
  - if `use_no_vig=True`, `market_prob_a` / `market_prob_b` are used as fair reference probabilities for edge screening
  - stake sizing and realized P&L still use the executable quoted price (`market_price_*`, direct market prob, or decimal odds)
- EV per $1 risked (yes contract price `c`, model win probability `p`):
  - `EV = p * (1/c - 1) - (1 - p)`
- Kelly fraction for binary contracts:
  - `f* = (p - c) / (1 - c)` (clamped to `>= 0`)
  - stake uses fractional Kelly via `kelly_fraction` and cap.
- ROI:
  - `roi = total_pnl / total_staked`
- Bankroll return:
  - `(ending_bankroll - initial_bankroll) / initial_bankroll`
- Max drawdown:
  - peak-to-trough drop on bankroll path.
- Sharpe convention:
  - per-bet Sharpe on **trade returns** (`pnl / stake`), sample std:
  - `sharpe = mean(returns) / std_sample(returns)`

## Input support

The engine supports these market input styles (in priority order):

1. Direct market probabilities (`market_prob_a`, `market_prob_b`)
2. Kalshi-style prices (`market_price_a`, `market_price_b`)
3. Decimal odds (`decimal_odds_a`, `decimal_odds_b`)

If both sides are available and `use_no_vig=True`, no-vig fair probabilities are used for edge comparison only; execution remains priced at the actual quoted contract.

## Bet policy

A bet is only placed when:

- edge passes `edge_threshold`
- model completeness passes `min_model_completeness` (if provided in row)
- confidence passes `min_confidence` (if configured and provided in row)
- market quality / liquidity passes configured checks

If both sides pass threshold and `allow_bet_on_both_sides=False`, ambiguous equal-edge cases are skipped.

Each row produces a structured `BetDecision`; executed bets also produce `TradeRecord`.

## Sizing modes

- `flat`
  - use `flat_stake` if set, otherwise `flat_fraction * bankroll`
- `fixed_fraction`
  - `fixed_bankroll_fraction * bankroll`
- `kelly`
  - fractional Kelly with `kelly_fraction` and `kelly_max_bankroll_fraction`

Global constraints:

- `min_stake`
- `max_stake` (or legacy `max_stake_abs`)

## Public API

```python
from deuce.backtest import BacktestConfig, run_backtest, summarize_backtest

rows = [
    {
        "event_id": "atp-miami-001",
        "matchup": "Player A vs Player B",
        "model_prob_a": 0.62,
        "market_price_a": 0.55,
        "market_price_b": 0.48,
        "outcome_a_won": 1,
        "data_completeness": 0.82,
        "market_quality": "ok",
    }
]

cfg = BacktestConfig(
    initial_bankroll=1000.0,
    edge_threshold=0.03,
    stake_mode="kelly",
    kelly_fraction=0.25,
    min_model_completeness=0.55,
)

result = run_backtest(rows, config=cfg)
summary = summarize_backtest(result)
print(summary["roi"], summary["sharpe_ratio"], summary["max_drawdown"])
```
