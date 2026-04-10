import pytest

from deuce.market.pricing import (
    MarketQuality,
    decimal_odds_from_prob,
    expected_value_per_dollar_risked,
    implied_prob_from_decimal_odds,
    kalshi_yes_price_to_prob,
    no_vig_two_outcome,
    normalize_two_sided_prices,
)


def test_kalshi_price():
    assert kalshi_yes_price_to_prob(0.62) == pytest.approx(0.62)
    assert kalshi_yes_price_to_prob(None) is None


def test_no_vig():
    a, b = no_vig_two_outcome(0.6, 0.5)
    assert a + b == pytest.approx(1.0)
    assert a > b


def test_normalize_two_sided():
    ia, ib, q = normalize_two_sided_prices(0.4, 0.5)
    assert ia + ib == pytest.approx(1.0)
    assert q == MarketQuality.OK


def test_decimal_implied_round_trip():
    p = 0.4
    o = decimal_odds_from_prob(p)
    assert implied_prob_from_decimal_odds(o) == pytest.approx(p)


def test_expected_value_per_dollar_risked():
    # At p=0.6 and price=0.5, each $1 risked has positive EV of +0.2.
    ev = expected_value_per_dollar_risked(0.6, 0.5)
    assert ev == pytest.approx(0.2)
