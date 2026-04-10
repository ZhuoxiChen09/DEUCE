"""Streamlit-facing formatting and market presentation helpers."""

from deuce.presentation.bet_signal import (
    DecisionRecord,
    bet_decision_summary,
    decision_record,
    decision_strip_html,
    route_reliability_label,
    status_badge_class,
)
from deuce.presentation.formatting import (
    UI_FAIL,
    UI_PASS,
    UI_WARN,
    checklist_to_ui,
    fair_odds_display,
    prob_to_american,
)

__all__ = [
    "DecisionRecord",
    "UI_FAIL",
    "UI_PASS",
    "UI_WARN",
    "bet_decision_summary",
    "checklist_to_ui",
    "decision_record",
    "decision_strip_html",
    "fair_odds_display",
    "prob_to_american",
    "route_reliability_label",
    "status_badge_class",
]
