"""Edge = model probability minus fair (or implied) market probability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeSignal:
    model_prob_a: float
    market_fair_a: float
    edge_a: float
    model_prob_b: float
    market_fair_b: float
    edge_b: float


def compute_edge(
    model_prob_a: float,
    market_prob_a: float,
    *,
    market_prob_b: float | None = None,
    use_fair: bool = True,
) -> EdgeSignal:
    """
    Edge on side A = model_prob_a - market_prob_a (by default ``market_prob_a`` should be fair / no-vig).

    If ``market_prob_b`` is omitted, it is ``1 - market_prob_a``.
    """
    mp_a = max(0.0, min(1.0, float(model_prob_a)))
    mkt_a = max(0.0, min(1.0, float(market_prob_a)))
    if market_prob_b is None:
        mkt_b = 1.0 - mkt_a
    else:
        mkt_b = max(0.0, min(1.0, float(market_prob_b)))
    if use_fair:
        s = mkt_a + mkt_b
        if s > 1e-9:
            fair_a, fair_b = mkt_a / s, mkt_b / s
        else:
            fair_a, fair_b = 0.5, 0.5
    else:
        fair_a, fair_b = mkt_a, mkt_b
    return EdgeSignal(
        model_prob_a=mp_a,
        market_fair_a=fair_a,
        edge_a=mp_a - fair_a,
        model_prob_b=1.0 - mp_a,
        market_fair_b=fair_b,
        edge_b=(1.0 - mp_a) - fair_b,
    )
