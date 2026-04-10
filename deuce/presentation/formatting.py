"""Probability ↔ odds helpers and shared terminal glyphs (no Streamlit)."""

# Pass / fail / warn glyphs used across Live, Bet, PvP, Research checklists.
UI_PASS = "✓"
UI_FAIL = "✗"
UI_WARN = "⚠"


def checklist_to_ui(text: str) -> str:
    """Replace PASS/FAIL/WARN tokens in checklist lines with shared terminal symbols."""
    return (
        text.replace("PASS", UI_PASS)
        .replace("FAIL", UI_FAIL)
        .replace("WARN", UI_WARN)
    )


def prob_to_american(p: float) -> str:
    """American odds string for win probability p in (0,1)."""
    if p <= 0 or p >= 1:
        return "—"
    if p >= 0.5:
        o = -100.0 * p / (1.0 - p)
        return f"{int(round(o))}"
    o = 100.0 * (1.0 - p) / p
    return f"+{int(round(o))}"


def fair_odds_display(p: float) -> str:
    """Decimal odds + American for display."""
    if p is None or p <= 0 or p >= 1:
        return "—"
    dec = 1.0 / p
    return f"{dec:.2f} ({prob_to_american(p)})"
