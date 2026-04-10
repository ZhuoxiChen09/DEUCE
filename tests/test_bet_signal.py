"""Decision logic: single source of truth for cards, PvP hygiene, bet board."""

from deuce.presentation.bet_signal import (
    bet_decision_summary,
    decision_record,
    route_reliability_label,
)


def test_route_reliability_label():
    assert route_reliability_label("core") == "ok"
    assert route_reliability_label("enhanced") == "ok"
    assert route_reliability_label("legacy") == "warn"
    assert route_reliability_label("fallback") == "warn"
    assert route_reliability_label("") == "warn"
    assert route_reliability_label("unknown") == "warn"


def _rec(**kw):
    base = dict(
        edge=0.04,
        abs_edge=0.04,
        completeness=0.7,
        market_quality="good",
        route="core",
        edge_reliable=True,
        low_information=False,
        market_valid=True,
        low_liquidity=False,
        min_edge=0.03,
        min_completeness=0.55,
    )
    base.update(kw)
    return decision_record(**base)


def test_decision_strong_bet_candidate():
    r = _rec()
    assert r.status == "bet_candidate"
    assert r.final_action == "BET CANDIDATE"
    assert r.edge_pass is True
    assert r.completeness_pass is True
    assert r.market_quality_pass is True
    assert r.route_reliability == "ok"


def test_positive_edge_low_completeness():
    r = _rec(completeness=0.4, abs_edge=0.08, edge=0.08)
    assert r.status == "low_confidence"
    assert r.completeness_pass is False


def test_positive_edge_weak_market_quality():
    r = _rec(market_quality="thin", abs_edge=0.08, edge=0.08)
    # Thin does not block bet_candidate in status logic, but mq flag fails
    assert r.market_quality_pass is False
    assert r.status == "bet_candidate"


def test_route_downgrade_legacy():
    r = _rec(route="legacy")
    assert r.route_reliability == "warn"
    assert r.status == "bet_candidate"


def test_no_edge():
    r = _rec(abs_edge=0.005, edge=0.005)
    assert r.status == "no_bet"
    assert r.edge_pass is False


def test_borderline_edge_watchlist():
    r = _rec(abs_edge=0.02, edge=0.02, min_edge=0.03)
    assert r.edge_pass is False
    assert r.status == "weak_edge"


def test_invalid_market():
    r = _rec(market_valid=False)
    assert r.status == "no_bet"


def test_low_liquidity():
    r = _rec(low_liquidity=True)
    assert r.status == "no_bet"


def test_not_edge_reliable():
    r = _rec(edge_reliable=False)
    assert r.status == "low_confidence"


def test_bet_decision_summary_matches_decision_record():
    kwargs = dict(
        edge=0.04,
        abs_edge=0.04,
        completeness=0.7,
        market_quality="good",
        route="core",
        edge_reliable=True,
        low_information=False,
        market_valid=True,
        low_liquidity=False,
        min_edge=0.03,
        min_completeness=0.55,
    )
    rec = decision_record(**kwargs)
    st, line, checks = bet_decision_summary(**kwargs)
    assert st == rec.status
    assert line == rec.one_line
    assert isinstance(checks, list)
    assert any("Edge" in c for c in checks)
    assert any("Route" in c for c in checks)
