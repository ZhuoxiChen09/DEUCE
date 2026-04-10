"""Developer / raw inspection tab (keep user-facing tabs clean)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from deuce.presentation.kpi import route_counts
from deuce.presentation.research_io import load_research_json


def render_developer_tab(
    matchups: Optional[List[Dict[str, Any]]],
    *,
    wf_path: str,
    core_path: Optional[str],
    enh_path: Optional[str],
    legacy_path: Optional[str],
    show_predictions: bool,
) -> None:
    st.caption("Internal inspection — not for trading decisions.")

    st.subheader("Bundle paths")
    st.code(
        "\n".join(
            [
                f"core:     {core_path or '—'}",
                f"enhanced: {enh_path or '—'}",
                f"legacy:   {legacy_path or '—'}",
            ]
        )
    )

    st.subheader("Research artifact")
    st.text(f"wf_json_path: {wf_path or '—'}")
    data = load_research_json(wf_path) if (wf_path or "").strip() else None
    st.metric("JSON loaded", "yes" if data else "no")
    if data:
        st.caption(f"Keys: {', '.join(sorted(data.keys())[:20])}{'…' if len(data) > 20 else ''}")

    st.subheader("Route allocation (current matchups)")
    if matchups:
        rc = route_counts(matchups)
        st.dataframe(pd.DataFrame([{"route": k, "n": v} for k, v in sorted(rc.items(), key=lambda x: -x[1])]), hide_index=True)
    else:
        st.caption("No matchups in session.")

    st.subheader("Active filter keys (session)")
    keys = [
        "term_sort",
        "term_min_edge",
        "term_min_comp",
        "term_route",
        "term_tour",
        "term_mq",
        "term_search",
        "term_bet_min_edge",
        "term_bet_min_comp",
        "show_predictions",
        "wf_json_path",
    ]
    fv = {k: st.session_state.get(k) for k in keys if k in st.session_state}
    st.json(fv)

    st.subheader("Enriched matchups (redacted JSON)")
    if matchups and show_predictions:
        slim: List[Dict[str, Any]] = []
        for m in matchups[:25]:
            slim.append(
                {
                    "event": m.get("event"),
                    "when": m.get("when"),
                    "deuce": m.get("deuce"),
                    "market_quality": m.get("market_quality"),
                    "players": m.get("players"),
                }
            )
        st.json(slim)
        if len(matchups) > 25:
            st.caption(f"Truncated to 25 of {len(matchups)} matchups.")
    else:
        st.caption("Load matchups to see rows.")

    with st.expander("Full session_state (debug)"):
        try:
            snap = {k: str(v)[:500] for k, v in st.session_state.items()}
            st.json(snap)
        except Exception as e:
            st.warning(str(e))
