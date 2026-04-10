"""Bet Candidates tab: ranked action board."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from deuce.presentation.kpi import (
    bet_board_rows,
    decision_counts_by_status,
    route_counts,
    sort_matchups_for_bet_board,
)
from deuce.presentation.streamlit_markets import render_rich_matchup_card


def render_bet_candidates_board(
    filtered: List[Dict[str, Any]],
    all_matchups: List[Dict[str, Any]],
    *,
    show_predictions: bool,
    min_edge_card: float,
    min_comp_card: float,
    live_auto_refresh: bool = True,
    live_refresh_seconds: int = 15,
) -> None:
    """Decision board: summary metrics, table, optional grouping, cards."""
    st.caption(
        f"**Pipeline:** {len(all_matchups)} loaded → **{len(filtered)}** pass bet-candidate filters "
        f"(reliable edge + completeness + liquidity + thresholds). "
        "**Rank score** = |edge| × route weight × market-quality weight + 0.12 × completeness "
        "(route **0.88** if legacy/fallback; MQ **0.65** if thin/malformed, **0.92** if unknown)."
    )
    counts = decision_counts_by_status(all_matchups, min_edge=min_edge_card, min_completeness=min_comp_card)
    rc = route_counts(filtered)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Bet candidate", counts.get("bet_candidate", 0))
    m2.metric("Watchlist", counts.get("weak_edge", 0))
    m3.metric("Low conf. / no bet", counts.get("low_confidence", 0) + counts.get("no_bet", 0))
    m4.metric("Shown (filtered)", len(filtered))
    m5.metric("Routes (shown)", len(rc))

    if rc:
        st.caption("Route mix (after filters): " + " · ".join(f"{k} **{v}**" for k, v in sorted(rc.items(), key=lambda x: -x[1])))

    ranked = sort_matchups_for_bet_board(filtered)
    rows = bet_board_rows(ranked, min_edge=min_edge_card, min_completeness=min_comp_card)
    if rows:
        df = pd.DataFrame(rows)
        show_cols = ["Event", "Event info", "Phase", "Matchup", "Score", "|edge|", "Route", "Compl.", "Mkt Q", "Action", "Edge", "Comp", "MQ"]
        df_show = df[[c for c in show_cols if c in df.columns]]
        st.subheader("Ranked board (composite score, then |edge|)")
        st.caption(
            "Rows sorted by **Score** (see pipeline caption). Expand **Structured checks** on cards for PASS/FAIL detail."
        )
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    compact = st.toggle("Compact card view", value=False, key="bet_compact_cards")
    group_by = st.selectbox("Group cards by", ["(none)", "route", "tour"], key="bet_group_cards")

    if group_by == "(none)":
        for m in ranked:
            render_rich_matchup_card(
                m,
                show_predictions=show_predictions,
                min_edge=min_edge_card,
                min_completeness=min_comp_card,
                compact=compact,
                live_auto_refresh=live_auto_refresh,
                live_refresh_seconds=live_refresh_seconds,
            )
        return

    if group_by == "route":
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for m in ranked:
            deuce = m.get("deuce") or {}
            r = str(deuce.get("route") or m.get("prediction_source") or "unknown")
            buckets.setdefault(r, []).append(m)
        for r in sorted(buckets.keys()):
            st.markdown(f"#### Route: **{r}**")
            for m in buckets[r]:
                render_rich_matchup_card(
                    m,
                    show_predictions=show_predictions,
                    min_edge=min_edge_card,
                    min_completeness=min_comp_card,
                    compact=compact,
                    live_auto_refresh=live_auto_refresh,
                    live_refresh_seconds=live_refresh_seconds,
                )
        return

    buckets2: Dict[str, List[Dict[str, Any]]] = {}
    for m in ranked:
        ev = str(m.get("event") or "")
        key = "WTA" if "wta" in ev.lower() else "ATP / other"
        buckets2.setdefault(key, []).append(m)
    for k in sorted(buckets2.keys()):
        st.markdown(f"#### {k}")
        for m in buckets2[k]:
            render_rich_matchup_card(
                m,
                show_predictions=show_predictions,
                min_edge=min_edge_card,
                min_completeness=min_comp_card,
                compact=compact,
                live_auto_refresh=live_auto_refresh,
                live_refresh_seconds=live_refresh_seconds,
            )
