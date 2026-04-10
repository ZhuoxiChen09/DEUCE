"""Human-readable summaries of terminal filter state (Streamlit session keys)."""

from __future__ import annotations

from typing import Any, Dict, List


def format_active_filters_summary(state: Dict[str, Any]) -> str:
    """One line + bullet list for display under headers."""
    parts: List[str] = []
    sort = state.get("term_sort", "edge")
    parts.append(f"sort={sort}")
    parts.append(f"|edge|≥{float(state.get('term_min_edge', 0)):.1%}")
    parts.append(f"compl≥{float(state.get('term_min_comp', 0)):.0%}")
    r = state.get("term_route", "all")
    if r != "all":
        parts.append(f"route={r}")
    t = state.get("term_tour", "all")
    if t != "all":
        parts.append(f"tour={t}")
    mq = state.get("term_mq", "all")
    if mq != "all":
        parts.append(f"mq={mq}")
    s = (state.get("term_search") or "").strip()
    if s:
        parts.append(f'search="{s}"')
    return " · ".join(parts)


def filter_diagnostics(
    *,
    n_total: int,
    n_after: int,
    n_if_edge_floor_removed: int | None = None,
) -> str:
    """Explain how many rows were filtered out."""
    dropped = n_total - n_after
    msg = f"**{n_after}** shown of **{n_total}** loaded (**{dropped}** hidden by active filters)."
    if n_if_edge_floor_removed is not None and n_if_edge_floor_removed != n_after:
        msg += f" With **min |edge| = 0**, **{n_if_edge_floor_removed}** would pass the other filters."
    return msg
