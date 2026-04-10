"""Bet / no-bet narrative from edge, coverage, and market quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from deuce.presentation.formatting import UI_FAIL, UI_PASS, UI_WARN, checklist_to_ui


@dataclass(frozen=True)
class DecisionRecord:
    """Structured pass/fail + action for UI (cards, PvP, bet board)."""

    status: str
    final_action: str
    edge_pass: bool
    completeness_pass: bool
    market_quality_pass: bool
    route_reliability: str
    one_line: str
    checks: Tuple[str, ...]


def route_reliability_label(route: str) -> str:
    r = (route or "").lower().strip()
    if r in ("core", "enhanced"):
        return "ok"
    if r in ("legacy",):
        return "warn"
    if r in ("fallback", ""):
        return "warn"
    return "warn"


def decision_record(
    *,
    edge: float,
    abs_edge: float,
    completeness: float | None,
    market_quality: str,
    route: str,
    edge_reliable: bool,
    low_information: bool,
    market_valid: bool,
    low_liquidity: bool,
    min_edge: float = 0.03,
    min_completeness: float = 0.55,
) -> DecisionRecord:
    """Single source of truth for bet_decision_summary + structured UI."""
    checks: List[str] = []
    ok_edge = abs_edge >= min_edge
    checks.append(f"Edge |{edge:.1%}| ≥ {min_edge:.0%}: {'PASS' if ok_edge else 'FAIL'}")
    ok_comp = completeness is None or completeness >= min_completeness
    if completeness is not None:
        checks.append(f"Completeness ≥ {min_completeness:.0%}: {'PASS' if ok_comp else 'FAIL'} ({completeness:.0%})")
    mq_raw = str(market_quality or "").lower()
    mq_ok = mq_raw not in ("thin", "malformed", "")
    if market_quality:
        checks.append(f"Market quality: {'PASS' if mq_ok else 'WARN'} ({market_quality or '—'})")
    else:
        checks.append("Market quality: unknown (not flagged)")
    rr = route_reliability_label(route)
    checks.append(f"Route reliability ({route or '—'}): {'OK' if rr == 'ok' else 'WARN'}")

    if not market_valid or low_liquidity:
        status = "no_bet"
        action = "NO BET"
        line = "No bet: market data is unreliable or illiquid."
    elif low_information or not edge_reliable:
        status = "low_confidence"
        action = "NO BET"
        line = "No bet: model or market context is too weak for a reliable edge."
    elif not ok_comp:
        status = "low_confidence"
        action = "NO BET"
        line = "No bet: completeness below threshold — edge is not trusted."
    elif ok_edge and ok_comp and edge_reliable and not low_information:
        status = "bet_candidate"
        action = "BET CANDIDATE"
        line = "Bet candidate: edge clears threshold and coverage is adequate vs fair no-vig price."
    elif abs_edge >= 0.015:
        status = "weak_edge"
        action = "WATCHLIST"
        line = "Watchlist: edge small or secondary checks failed."
    else:
        status = "no_bet"
        action = "NO BET"
        line = "No bet: edge vs fair price is within noise."

    if not market_valid:
        checks.append("Market pairing invalid — do not trust prices.")
    if low_liquidity:
        checks.append("Liquidity thin — price may not be tradeable.")

    return DecisionRecord(
        status=status,
        final_action=action,
        edge_pass=ok_edge,
        completeness_pass=ok_comp,
        market_quality_pass=mq_ok,
        route_reliability=rr,
        one_line=line,
        checks=tuple(checks),
    )


def bet_decision_summary(
    *,
    edge: float,
    abs_edge: float,
    completeness: float | None,
    market_quality: str,
    route: str,
    edge_reliable: bool,
    low_information: bool,
    market_valid: bool,
    low_liquidity: bool,
    min_edge: float = 0.03,
    min_completeness: float = 0.55,
) -> Tuple[str, str, List[str]]:
    """
    Returns (status_label, one_line_summary, checklist_strings).

    status_label: "bet_candidate" | "weak_edge" | "no_bet" | "low_confidence"
    """
    rec = decision_record(
        edge=edge,
        abs_edge=abs_edge,
        completeness=completeness,
        market_quality=market_quality,
        route=route,
        edge_reliable=edge_reliable,
        low_information=low_information,
        market_valid=market_valid,
        low_liquidity=low_liquidity,
        min_edge=min_edge,
        min_completeness=min_completeness,
    )
    legacy = [checklist_to_ui(c) for c in rec.checks]
    if route:
        legacy = [x for x in legacy if not x.startswith("Route reliability")]
        legacy.append(f"Route: **{route}**")
    return rec.status, rec.one_line, legacy


def status_badge_class(status: str) -> str:
    return {
        "bet_candidate": "tp-badge-high",
        "weak_edge": "tp-badge-med",
        "no_bet": "tp-badge-low",
        "low_confidence": "tp-badge-low",
    }.get(status, "tp-badge-low")


def decision_strip_html(rec: DecisionRecord) -> str:
    """HTML pass/fail strip for cards (Streamlit markdown unsafe_allow_html)."""
    e_ok = rec.edge_pass
    c_ok = rec.completeness_pass
    mq_ok = rec.market_quality_pass
    rr_ok = rec.route_reliability == "ok"
    return f"""
<div style="display:flex;flex-wrap:wrap;gap:0.35rem;align-items:center;margin:0.4rem 0 0.6rem 0;font-size:0.78rem;">
  <span class="tp-badge {'tp-badge-high' if e_ok else 'tp-badge-low'}">EDGE {UI_PASS if e_ok else UI_FAIL}</span>
  <span class="tp-badge {'tp-badge-high' if c_ok else 'tp-badge-low'}">COMP {UI_PASS if c_ok else UI_FAIL}</span>
  <span class="tp-badge {'tp-badge-high' if mq_ok else 'tp-badge-med'}">MQ {UI_PASS if mq_ok else UI_WARN}</span>
  <span class="tp-badge {'tp-badge-high' if rr_ok else 'tp-badge-med'}">ROUTE {"OK" if rr_ok else "WARN"}</span>
  <span class="tp-badge {status_badge_class(rec.status)}">{rec.final_action}</span>
</div>
"""
