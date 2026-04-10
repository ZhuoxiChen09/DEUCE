"""Streamlit components for DEUCE market terminal (import streamlit only here)."""

from __future__ import annotations

import html
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from api import parse_iso8601_utc
from deuce.live_scores import fetch_scoreboard_by_dates, resolve_live_score_for_matchup
from deuce.presentation.bet_signal import decision_record, decision_strip_html, status_badge_class
from deuce.presentation.formatting import checklist_to_ui, fair_odds_display, prob_to_american
from deuce.presentation.live_scoreboard_utils import live_score_source_label, split_score_tokens


def render_kpi_strip(stats: Dict[str, Any], bundle_one_liner: str) -> None:
    """Seven KPI tiles + bundle summary line."""
    c = st.columns([1, 1, 1, 1, 1, 1, 2])
    labels = [
        ("Markets", str(stats.get("n_markets", "-"))),
        ("Priced", str(stats.get("n_priced", "-"))),
        ("Edge >2%", str(stats.get("n_positive_edge", "-"))),
        ("Avg |edge|", f"{stats['avg_edge']:.1%}" if stats.get("avg_edge") is not None else "-"),
        ("Max |edge|", f"{stats['max_edge']:.1%}" if stats.get("max_edge") is not None else "-"),
        ("Avg compl.", f"{stats['avg_completeness']:.0%}" if stats.get("avg_completeness") is not None else "-"),
        ("Route mix", _route_mix(stats)),
    ]
    for i, (lab, val) in enumerate(labels):
        with c[i]:
            st.metric(lab, val)
    st.caption(bundle_one_liner)


def _route_mix(stats: Dict[str, Any]) -> str:
    if stats.get("pct_core") is None:
        return "-"
    return (
        f"C {stats['pct_core']:.0%} | E {stats['pct_enhanced']:.0%} | "
        f"L/FB {stats['pct_legacy_or_fb']:.0%}"
    )


def render_rich_matchup_card(
    m: Dict[str, Any],
    *,
    show_predictions: bool,
    min_edge: float,
    min_completeness: float,
    compact: bool = False,
    live_auto_refresh: bool = True,
    live_refresh_seconds: int = 15,
) -> None:
    """Single matchup card with prices, scoreboard, and model context."""
    title = m.get("event") or m.get("title") or "Market"
    when = m.get("when_display") or m.get("when") or ""
    scheduled_start = m.get("scheduled_start_display") or m.get("scheduled_start") or when
    event_info_line = str(m.get("event_info_line") or "").strip()
    phase_label = str(m.get("market_phase_label") or "").strip()
    phase_detail = str(m.get("market_phase_detail") or "").strip()
    live_score = m.get("live_score") or {}
    players = m.get("players") or []
    deuce = m.get("deuce") or {}
    route = str(deuce.get("route") or m.get("prediction_source") or "unknown")
    badge_route = {
        "core": ("CORE", "tp-badge-med"),
        "enhanced": ("ENHANCED", "tp-badge-high"),
        "legacy": ("LEGACY", "tp-badge-low"),
        "fallback": ("FALLBACK", "tp-badge-low"),
    }.get(route.lower(), (route.upper()[:10], "tp-badge-low"))
    phase_badge = {
        "pregame": ("PREGAME", "tp-badge-high"),
        "live": ("LIVE", "tp-badge-high"),
        "completed": ("FINAL", "tp-badge-low"),
        "started_or_live_like": ("STARTED?", "tp-badge-med"),
        "unknown": ("START?", "tp-badge-low"),
    }.get(str(m.get("market_phase") or "").lower(), (phase_label.upper()[:12] or "START?", "tp-badge-low"))

    has_two_players = len(players) == 2
    has_predictions = show_predictions and has_two_players
    mvalid = bool(m.get("market_valid", True))
    low_liq = bool(m.get("low_liquidity", False))
    mq = str(m.get("market_quality") or "")
    edge_rel = bool(m.get("edge_reliable", False))
    low_info = bool(m.get("low_information_prediction", False))

    p0 = players[0] if len(players) > 0 else {}
    p1 = players[1] if len(players) > 1 else {}
    e0 = p0.get("edge")
    e1 = p1.get("edge")
    ae0 = abs(float(e0)) if e0 is not None else 0.0
    ae1 = abs(float(e1)) if e1 is not None else 0.0
    lead = 0 if ae0 >= ae1 else 1
    edge = float(e0) if lead == 0 and e0 is not None else float(e1 or 0.0)
    abs_edge = max(ae0, ae1)
    comp = deuce.get("data_completeness")
    comp_f = float(comp) if comp is not None else None

    if has_predictions:
        rec = decision_record(
            edge=edge,
            abs_edge=abs_edge,
            completeness=comp_f,
            market_quality=mq,
            route=route,
            edge_reliable=edge_rel,
            low_information=low_info,
            market_valid=mvalid,
            low_liquidity=low_liq,
            min_edge=min_edge,
            min_completeness=min_completeness,
        )
        status = rec.status
        summary = rec.one_line
        checks = [checklist_to_ui(c) for c in rec.checks]
        badge_cls = status_badge_class(status)
        status_label = {
            "bet_candidate": "BET CANDIDATE",
            "weak_edge": "WATCHLIST",
            "no_bet": "NO BET",
            "low_confidence": "LOW CONFIDENCE",
        }.get(status, status.upper())
    else:
        rec = None
        summary = ""
        checks = []
        status = "info"
        badge_cls = "tp-badge-low"
        status_label = "LIVE VIEW"

    border_color = "#22C55E" if status == "bet_candidate" else "#eab308" if status == "weak_edge" else "#64748b"
    hdr = f"""
<div class="tp-card" style="border-left: 4px solid {border_color};">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:0.75rem;flex-wrap:wrap;">
    <div>
      <div class="tp-card-header" style="margin-bottom:0.2rem;">{title}</div>
      <div style="font-size:0.85rem;opacity:0.85;">{scheduled_start}</div>
      {"<div style='font-size:0.82rem;opacity:0.8;margin-top:0.25rem;'>" + event_info_line + "</div>" if event_info_line else ""}
    </div>
    <div style="display:flex;gap:0.5rem;align-items:center;">
      <span class="tp-badge {phase_badge[1]}">{phase_badge[0]}</span>
      <span class="tp-badge {badge_route[1]}">{badge_route[0]}</span>
      <span class="tp-badge {badge_cls}">{status_label}</span>
    </div>
  </div>
</div>
"""
    st.markdown(hdr, unsafe_allow_html=True)

    if has_two_players:
        photo_cols = st.columns(2)
        for idx, player in enumerate(players[:2]):
            with photo_cols[idx]:
                image_url = player.get("image_url")
                if image_url:
                    st.image(image_url, width=88)
                label = str(player.get("name") or "")
                profile_url = str(player.get("profile_url") or "").strip()
                if profile_url:
                    st.markdown(f"**[{label}]({profile_url})**")
                else:
                    st.markdown(f"**{label}**")

    info_col, score_col = st.columns([1.1, 1.0])
    with info_col:
        if has_two_players:
            lead_player = players[lead]
            lead_name = str(lead_player.get("name") or "")
            lead_price = lead_player.get("yes_bid")
            if lead_name and lead_price is not None:
                st.markdown(f"**Entry price:** Buy YES on {lead_name} at ${float(lead_price):.2f} per share")
            else:
                st.markdown("**Entry price:** Unavailable")
            alt_player = players[1 - lead]
            alt_name = str(alt_player.get("name") or "")
            alt_price = alt_player.get("yes_bid")
            if alt_name and alt_price is not None:
                st.caption(f"Other side: {alt_name} at ${float(alt_price):.2f} per share")
        else:
            st.markdown("**Entry price:** Unavailable")
        if not has_predictions:
            st.caption("Load predictions to see model edge and fair no-vig price.")
    with score_col:
        render_live_scoreboard_panel(
            m,
            matchup_players=players,
            auto_refresh=live_auto_refresh,
            refresh_seconds=live_refresh_seconds,
        )

    if has_predictions and rec is not None:
        st.markdown(decision_strip_html(rec), unsafe_allow_html=True)
    if phase_detail:
        source_hint = (
            f"from {live_score_source_label(str(live_score.get('source') or ''))}"
            if live_score.get("matched")
            else "inferred from Kalshi timing"
        )
        st.caption(f"Market state: {phase_label or phase_badge[0]} - {phase_detail} ({source_hint})")

    rows = []
    for p in players:
        mp = p.get("model_prob")
        mkt = p.get("market_prob", p.get("kalshi_prob"))
        fair = p.get("market_fair_prob")
        ed = p.get("edge")
        yb = p.get("yes_bid")
        row = {
            "Player": p.get("name", ""),
            "Kalshi bid ($)": f"${float(yb):.2f}" if yb is not None else "-",
            "Model": f"{mp:.1%}" if mp is not None else "-",
            "Market impl.": f"{mkt:.1%}" if mkt is not None else "-",
            "Mkt American": prob_to_american(float(mkt)) if mkt is not None else "-",
            "Fair (no-vig)": f"{fair:.1%}" if fair is not None else "-",
            "Fair odds": fair_odds_display(float(fair)) if fair is not None else "-",
            "Edge vs fair": f"{ed:+.1%}" if ed is not None else "-",
        }
        if compact and has_predictions:
            row = {k: row[k] for k in ("Player", "Model", "Market impl.", "Edge vs fair")}
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not has_predictions:
        st.divider()
        return

    ev = m.get("evaluation")
    conf = deuce.get("evidence_quality") or "-"
    lb = deuce.get("leaderboard_coverage_score")
    parts = []
    if comp_f is not None:
        parts.append(f"completeness **{comp_f:.0%}**")
    if lb is not None:
        parts.append(f"leaderboard coverage **{lb:.2f}**")
    parts.append(f"evidence **{conf}**")
    st.markdown("**Trust:** " + " | ".join(parts))

    st.markdown(f"**Decision:** {summary}")
    with st.expander("Structured checks & drivers", expanded=False):
        st.markdown("**Checks**")
        for line in checks:
            st.markdown(f"- {line}")
        if ev and getattr(ev, "feature_breakdown", None):
            bd = ev.feature_breakdown
            top = sorted(bd.items(), key=lambda x: -abs(x[1]))[:6]
            st.markdown("**Top grouped drivers (model vs market context)**")
            st.dataframe(
                pd.DataFrame(top, columns=["Bucket", "Contribution"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(getattr(ev, "explanation", ""))
    warns = deuce.get("warnings") or []
    for w in warns:
        st.warning(w)
    st.divider()


def render_model_diagnostics_panel(
    core_ok: bool,
    core_path: Optional[str],
    meta: Optional[Dict[str, Any]],
) -> None:
    """Formatted Core bundle / training state (not raw JSON first)."""
    st.subheader("Quant diagnostics | Core bundle")
    if not core_ok or not core_path:
        st.warning("Core bundle path missing or not found. Train or place **artifacts/tennis_model_core.pkl** (or configured path).")
        return
    if not meta:
        st.info("Could not read bundle metadata.")
        return
    if meta.get("error"):
        st.error("Bundle load error: **%s**" % meta.get("error"))
        return
    if meta.get("format") == "legacy":
        st.warning("Legacy pickle format - limited metadata. Retrain for v2/v3 bundle fields.")
        rows = [("Path", core_path), ("Features (approx.)", meta.get("n_features"))]
    else:
        rows = [
            ("Active path", core_path),
            ("Bundle format", meta.get("format")),
            ("Model tier", meta.get("model_tier")),
            ("Feature set", meta.get("feature_set_name")),
            ("Feature count", meta.get("n_features")),
            ("Split strategy", meta.get("split_strategy")),
            ("Calibration method", meta.get("calibration_method")),
            ("Val log loss (calibrated)", meta.get("val_log_loss_calibrated")),
            ("Train window", str(meta.get("train_date_window"))),
            ("Validation window", str(meta.get("validation_date_window"))),
            ("Test window", str(meta.get("test_date_window"))),
            ("Samples total", meta.get("n_samples_total")),
            ("n_train / n_val / n_test", "%s / %s / %s" % (meta.get("n_train"), meta.get("n_val"), meta.get("n_test_holdout"))),
            ("Code version", meta.get("code_version")),
            ("Training timestamp", meta.get("created_at")),
        ]
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("Raw bundle JSON"):
        st.json(meta)


def render_live_scoreboard_panel(
    matchup: Dict[str, Any],
    *,
    matchup_players: Optional[List[Dict[str, Any]]] = None,
    auto_refresh: bool,
    refresh_seconds: int,
) -> None:
    """Render only the live-score area as a fragment-scoped refresh region."""
    interval = f"{max(1, int(refresh_seconds))}s" if auto_refresh and _should_poll_scoreboard(matchup) else None

    @st.fragment(run_every=interval)
    def _panel() -> None:
        live_score = _refresh_live_score(matchup)
        is_live = bool(live_score.get("matched")) and bool(live_score.get("is_live"))
        if live_score.get("matched"):
            st.markdown("**Live scoreboard**")
            st.markdown(
                _live_scoreboard_html(
                    live_score,
                    matchup_players=matchup_players or [],
                    refresh_seconds=max(1, int(refresh_seconds)),
                    auto_refresh=auto_refresh,
                    is_live=is_live,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**Live scoreboard:** Not matched yet")

    _panel()


@st.cache_data(show_spinner=False, ttl=2)
def _cached_fragment_scoreboard(date_keys: tuple[str, ...]) -> List[Dict[str, Any]]:
    return fetch_scoreboard_by_dates(list(date_keys))


def _matchup_local_date_key(matchup: Dict[str, Any]) -> str:
    dt = parse_iso8601_utc(matchup.get("scheduled_start") or matchup.get("when"))
    if dt is None:
        return ""
    return dt.astimezone().strftime("%Y%m%d")


def _should_poll_scoreboard(matchup: Dict[str, Any]) -> bool:
    live_score = matchup.get("live_score") if isinstance(matchup.get("live_score"), dict) else {}
    if live_score.get("is_completed"):
        return False
    local_date = _matchup_local_date_key(matchup)
    if not local_date:
        return bool(live_score.get("is_live"))
    today_key = datetime.now().astimezone().strftime("%Y%m%d")
    return local_date <= today_key


def _refresh_live_score(matchup: Dict[str, Any]) -> Dict[str, Any]:
    local_date = _matchup_local_date_key(matchup)
    scoreboard = _cached_fragment_scoreboard((local_date,)) if local_date else []
    return resolve_live_score_for_matchup(matchup, scoreboard_matches=scoreboard)


def _merge_scoreboard_players(
    score_players: List[Dict[str, Any]],
    matchup_players: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_name = {str(p.get("name") or "").strip().lower(): p for p in matchup_players}
    for player in score_players:
        row = dict(player)
        ref = by_name.get(str(player.get("name") or "").strip().lower())
        if ref:
            if ref.get("image_url"):
                row["image_url"] = ref.get("image_url")
            if ref.get("profile_url"):
                row["profile_url"] = ref.get("profile_url")
        out.append(row)
    return out


def _live_scoreboard_html(
    live_score: Dict[str, Any],
    *,
    matchup_players: List[Dict[str, Any]],
    refresh_seconds: int,
    auto_refresh: bool,
    is_live: bool,
) -> str:
    players = _merge_scoreboard_players(list(live_score.get("players") or []), matchup_players)
    split_rows = [split_score_tokens(list(p.get("score_tokens") or []), is_live=is_live) for p in players]
    max_sets = max((len(sets) for sets, _ in split_rows), default=0)
    has_live_points = any(live_point is not None for _, live_point in split_rows)
    status_text = html.escape(str(live_score.get("status_text") or live_score.get("summary") or "Matched"))
    venue_bits = [str(live_score.get("tournament") or ""), str(live_score.get("round") or ""), str(live_score.get("court") or "")]
    venue_text = html.escape(" | ".join(bit for bit in venue_bits if bit))
    source_label = html.escape(live_score_source_label(str(live_score.get("source") or "")))

    header_cells = "".join(
        f'<div style="min-width:22px;text-align:center;color:#94a3b8;font-size:0.76rem;font-weight:700;">S{i + 1}</div>'
        for i in range(max_sets)
    )
    pts_header = '<div style="min-width:30px;text-align:center;color:#94a3b8;font-size:0.76rem;font-weight:700;">Pts</div>' if has_live_points else ""

    rows_html: List[str] = []
    for player, (set_scores, live_point) in zip(players, split_rows):
        name = html.escape(str(player.get("name") or ""))
        image_url = str(player.get("image_url") or "").strip()
        profile_url = str(player.get("profile_url") or "").strip()
        label = f'<a href="{html.escape(profile_url)}" target="_blank">{name}</a>' if profile_url else name
        thumb = (
            f'<img src="{html.escape(image_url)}" style="width:26px;height:26px;border-radius:999px;object-fit:cover;border:1px solid #334155;" />'
            if image_url
            else '<div style="width:26px;height:26px;border-radius:999px;background:#1e293b;border:1px solid #334155;"></div>'
        )
        set_cells = "".join(
            f'<div style="min-width:22px;text-align:center;font-weight:600;color:#E5E7EB;">{html.escape(score)}</div>'
            for score in set_scores
        )
        if len(set_scores) < max_sets:
            set_cells += "".join(
                '<div style="min-width:22px;text-align:center;color:#64748b;">-</div>'
                for _ in range(max_sets - len(set_scores))
            )
        live_cell = (
            f'<div style="min-width:30px;height:24px;border:1px solid #86efac;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#86efac;font-weight:700;background:rgba(34,197,94,0.08);">{html.escape(live_point)}</div>'
            if live_point is not None
            else ('<div style="min-width:30px;height:24px;"></div>' if has_live_points else "")
        )
        rows_html.append(
            (
                f'<div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:0.6rem;align-items:center;padding:0.55rem 0.65rem;border-top:1px solid #223046;">'
                f'<div style="display:flex;align-items:center;gap:0.55rem;min-width:0;">'
                f"{thumb}"
                f'<div style="min-width:0;color:#F8FAFC;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>'
                f"</div>"
                f'<div style="display:flex;gap:0.55rem;align-items:center;">{set_cells}</div>'
                f"{live_cell}"
                f"</div>"
            )
        )

    return textwrap.dedent(
        f"""
        <div style="border:1px solid #223046;border-radius:10px;overflow:hidden;background:#0f172a;">
          <div style="display:flex;justify-content:space-between;gap:0.75rem;align-items:center;padding:0.55rem 0.65rem;background:#111c2f;border-bottom:1px solid #223046;">
            <div style="color:#E5E7EB;font-weight:700;">{status_text}</div>
            <div style="color:{'#86efac' if is_live else '#94a3b8'};font-weight:700;">{'LIVE' if is_live else 'FINAL' if live_score.get('is_completed') else 'SCHEDULED'}</div>
          </div>
          <div style="display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:0.6rem;align-items:center;padding:0.4rem 0.65rem;background:#10192a;border-bottom:1px solid #223046;">
            <div style="color:#94a3b8;font-size:0.76rem;font-weight:700;">Player</div>
            <div style="display:flex;gap:0.55rem;align-items:center;">{header_cells}</div>
            {pts_header}
          </div>
          {''.join(rows_html)}
          <div style="padding:0.5rem 0.65rem;border-top:1px solid #223046;color:#94a3b8;font-size:0.83rem;">{venue_text or source_label}</div>
        </div>
        """
    ).strip()


def render_live_market_charts(matchups: List[Dict[str, Any]], *, show_predictions: bool) -> None:
    """Focused charts: edge, completeness, route, model vs market (when data exists)."""
    if not show_predictions or not matchups:
        return
    edges: List[float] = []
    routes: List[str] = []
    comps: List[float] = []
    mprob: List[float] = []
    mkprob: List[float] = []
    for m in matchups:
        deuce = m.get("deuce") or {}
        r = str(deuce.get("route") or m.get("prediction_source") or "unknown")
        routes.append(r)
        c = deuce.get("data_completeness")
        if c is not None:
            comps.append(float(c))
        for p in m.get("players") or []:
            e = p.get("edge")
            if e is not None:
                edges.append(abs(float(e)))
            mp = p.get("model_prob")
            mkt = p.get("market_prob", p.get("kalshi_prob"))
            if mp is not None and mkt is not None:
                mprob.append(float(mp))
                mkprob.append(float(mkt))
    st.markdown("**Distribution**")
    c1, c2, c3 = st.columns(3)
    if len(edges) >= 3:
        with c1:
            st.caption("|edge| (per quoted side)")
            st.bar_chart(pd.DataFrame({"abs_edge": edges}))
    if comps:
        with c2:
            st.caption("Data completeness")
            st.bar_chart(pd.DataFrame({"completeness": comps}))
    if routes and len(set(routes)) >= 1:
        vc = pd.Series(routes).value_counts()
        with c3:
            st.caption("Route allocation (matchups)")
            st.bar_chart(vc.to_frame("count"))
    if len(mprob) >= 3:
        st.caption("Model vs market implied probability (per side)")
        st.scatter_chart(pd.DataFrame({"model_p": mprob, "market_p": mkprob}))


def build_bundle_summary_line(core_ok: bool, core_path: Optional[str], meta: Optional[Dict[str, Any]]) -> str:
    if not core_ok or not meta or meta.get("format") == "legacy":
        return "Bundle: **legacy or missing** - train with `python -m deuce.cli train` for v3 metadata."
    return (
        f"Active Core bundle: **{meta.get('model_tier', '?')}** | cal **{meta.get('calibration_method', '?')}** | "
        f"split **{meta.get('split_strategy', '?')}** | val LL **{meta.get('val_log_loss_calibrated', '-')}**"
    )
