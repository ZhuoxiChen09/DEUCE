"""Flat and fractional Kelly stake sizing for binary contracts."""

from __future__ import annotations


def flat_stake(bankroll: float, fraction: float = 0.01) -> float:
    """Risk ``fraction`` of bankroll (default 1%)."""
    return max(0.0, float(bankroll) * max(0.0, float(fraction)))


def fractional_kelly_stake(
    bankroll: float,
    model_prob: float,
    market_price: float,
    *,
    kelly_fraction: float = 0.25,
    max_fraction_of_bankroll: float = 0.05,
) -> float:
    """
    Binary contract Kelly: f* = (p - cost) / (1 - cost) for yes at price ``market_price``.

    ``market_price`` is the contract price in [0,1] (what you pay per $1 payout).
    Stake is capped at ``max_fraction_of_bankroll`` * bankroll.
    """
    p = max(1e-9, min(1.0 - 1e-9, float(model_prob)))
    c = max(1e-9, min(1.0 - 1e-9, float(market_price)))
    if p <= c:
        return 0.0
    f_star = (p - c) / (1.0 - c)
    f = max(0.0, f_star * float(kelly_fraction))
    cap = float(bankroll) * max_fraction_of_bankroll
    return min(f * float(bankroll), cap)
