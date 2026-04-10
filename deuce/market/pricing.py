"""
Kalshi-style market pricing: implied probabilities, two-sided normalization, no-vig fair prices.

Kept UI-agnostic so CLI, Streamlit, and backtests share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class MarketQuality(str, Enum):
    """Coarse liquidity / completeness label for a two-outcome contract."""

    OK = "ok"
    THIN = "thin"  # both sides very low — unreliable
    ONE_SIDED = "one_sided"  # only one side has a usable quote
    MALFORMED = "malformed"  # NaN, negative, or missing both


def decimal_odds_from_prob(prob: float) -> float:
    """Decimal odds payout multiple from implied probability in (0,1)."""
    p = max(1e-9, min(1.0 - 1e-9, float(prob)))
    return 1.0 / p


def implied_prob_from_decimal_odds(decimal_odds: float) -> float:
    """Implied probability from decimal odds > 1."""
    o = max(1.0000001, float(decimal_odds))
    return 1.0 / o


def expected_value_per_dollar_risked(model_prob: float, market_prob: float) -> float:
    """
    EV per $1 risked for a binary yes contract priced at ``market_prob``.

    If you pay c dollars for $1 payout and win probability is p:
      EV($1 risked) = p * (1/c - 1) - (1 - p)
    """
    p = max(0.0, min(1.0, float(model_prob)))
    c = max(1e-9, min(1.0 - 1e-9, float(market_prob)))
    return p * ((1.0 / c) - 1.0) - (1.0 - p)


def kalshi_yes_price_to_prob(yes_price: Any) -> Optional[float]:
    """
    Convert Kalshi yes price (dollars per share, e.g. 0.62) to implied probability in [0, 1].
    Returns None if missing or invalid.
    """
    if yes_price is None:
        return None
    try:
        v = float(yes_price)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return max(0.0, min(1.0, v))


def normalize_two_sided_prices(
    price_a: Any,
    price_b: Any,
    *,
    thin_threshold: float = 0.05,
) -> Tuple[float, float, MarketQuality]:
    """
    Turn two yes-bid prices into normalized probabilities that sum to 1 when both are present.

    If both raw prices are below ``thin_threshold``, quality is THIN.
    If exactly one side is valid, the other gets the remainder (ONE_SIDED).
    """
    pa = kalshi_yes_price_to_prob(price_a)
    pb = kalshi_yes_price_to_prob(price_b)
    if pa is None and pb is None:
        return 0.5, 0.5, MarketQuality.MALFORMED
    if pa is None:
        pb = pb or 0.0
        return max(0.0, min(1.0, 1.0 - pb)), max(0.0, min(1.0, pb)), MarketQuality.ONE_SIDED
    if pb is None:
        pa = pa or 0.0
        return max(0.0, min(1.0, pa)), max(0.0, min(1.0, 1.0 - pa)), MarketQuality.ONE_SIDED
    total = pa + pb
    if total <= 0:
        return 0.5, 0.5, MarketQuality.MALFORMED
    na, nb = pa / total, pb / total
    if pa < thin_threshold and pb < thin_threshold:
        return na, nb, MarketQuality.THIN
    return na, nb, MarketQuality.OK


def no_vig_two_outcome(p_a: float, p_b: float) -> Tuple[float, float]:
    """
    Remove book margin assuming binary market: p_a + p_b may exceed 1.

    Returns fair probabilities (f_a, f_b) that sum to 1. If sum <= 0, returns (0.5, 0.5).
    """
    s = float(p_a) + float(p_b)
    if s <= 1e-12:
        return 0.5, 0.5
    return float(p_a) / s, float(p_b) / s


@dataclass(frozen=True)
class TwoSidedQuote:
    """Raw and processed two-outcome market view."""

    raw_a: Optional[float]
    raw_b: Optional[float]
    implied_a: float
    implied_b: float
    fair_a: float
    fair_b: float
    quality: MarketQuality

    @classmethod
    def from_yes_bids(
        cls,
        yes_a: Any,
        yes_b: Any,
        *,
        thin_threshold: float = 0.05,
    ) -> "TwoSidedQuote":
        ra = kalshi_yes_price_to_prob(yes_a)
        rb = kalshi_yes_price_to_prob(yes_b)
        ia, ib, q = normalize_two_sided_prices(yes_a, yes_b, thin_threshold=thin_threshold)
        if ra is not None and rb is not None:
            fa, fb = no_vig_two_outcome(ra, rb)
        else:
            fa, fb = ia, ib
        return cls(
            raw_a=ra,
            raw_b=rb,
            implied_a=ia,
            implied_b=ib,
            fair_a=fa,
            fair_b=fb,
            quality=q,
        )
