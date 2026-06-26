"""Archived legacy Streamlit interface; retained for explicit compatibility use."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from cli import (
    ATP_TDI_TIME_FRAMES,
    ATP_WINLOSS_INDEX_CATEGORIES,
    ATP_WINLOSS_TIME_CATEGORIES,
    enrich_matchups_with_predictions,
    get_atp_rankings_display,
    get_atp_pressure_leaderboard_display,
    get_atp_return_leaderboard_display,
    get_atp_serve_leaderboard_display,
    get_atp_tdi_leaderboard_display,
    get_atp_winloss_index_display,
    get_tennis_matchups,
)
from deuce.player_db import DEFAULT_DB_PATH, load_db, update_db_from_atp_datasets

from model_predict import (
    get_default_core_path,
    get_default_enhanced_path,
    get_default_model_path,
    load_model,
)

from deuce.presentation.bet_board import render_bet_candidates_board
from deuce.presentation.developer_tab import render_developer_tab
from deuce.presentation.filters import filter_diagnostics, format_active_filters_summary
from deuce.presentation.kpi import filter_matchups, summarize_matchups_kpis
from deuce.presentation.model_ops import render_model_operations_panel
from deuce.presentation.pvp_lab import render_pvp_lab
from deuce.presentation.research_schema import OOS_CSV_COLUMNS_HELP, WALK_FORWARD_JSON_FIELDS_HELP
from deuce.presentation.research_view import render_research_tab
from deuce.presentation.streamlit_markets import (
    build_bundle_summary_line,
    render_kpi_strip,
    render_live_market_charts,
    render_rich_matchup_card,
)

st.set_page_config(page_title="DEUCE — Tennis market terminal", layout="wide")

LIVE_SCORE_REFRESH_SECONDS = 2
KALSHI_MATCHUPS_REFRESH_SECONDS = 15


@st.cache_data(show_spinner=False)
def _cached_walk_forward_json(path: str, mtime: float) -> Optional[Dict[str, Any]]:
    from deuce.presentation.research_io import load_research_json

    return load_research_json(path)


@st.cache_data(show_spinner=False, ttl=2)
def _cached_live_scoreboard(date_keys: tuple[str, ...]) -> List[Dict[str, Any]]:
    from deuce.live_scores import fetch_scoreboard_by_dates

    return fetch_scoreboard_by_dates(list(date_keys))


@st.cache_data(show_spinner=False, ttl=KALSHI_MATCHUPS_REFRESH_SECONDS)
def _cached_kalshi_matchups(player: str, days: int, max_results: int) -> Tuple[List[Dict[str, Any]], str]:
    return get_tennis_matchups(
        player=player.strip() or None,
        days=days,
        max_results=max_results,
        fast=True,
    )


st.markdown(
    """
    <style>
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: #0B1220 !important;
        color: #F9FAFB !important;
    }
    .block-container { background: #0B1220 !important; color: #F9FAFB !important; }
    section[data-testid="stSidebar"] {
        background: #0F172A !important;
        border-right: 1px solid #1F2937;
    }
    input, textarea, select {
        background: #111827 !important;
        color: #F9FAFB !important;
        border: 1px solid #1F2937 !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 12px !important;
    }
    .tp-card {
        background: #111827;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border: 1px solid #1F2937;
        margin-bottom: 0.75rem;
    }
    .tp-card-header {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9CA3AF;
    }
    .tp-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.08rem 0.5rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .tp-badge-high { background: rgba(34, 197, 94, 0.12); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.5); }
    .tp-badge-med { background: rgba(234, 179, 8, 0.12); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.6); }
    .tp-badge-low { background: rgba(148, 163, 184, 0.18); color: #9CA3AF; border: 1px solid rgba(148, 163, 184, 0.6); }
    .tp-bar-row { display: flex; align-items: center; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.25rem; }
    .tp-bar-track { flex: 1; height: 6px; border-radius: 999px; margin: 0 0.5rem; background: #1F2937; overflow: hidden; position: relative; }
    .tp-bar-fill-model { position: absolute; inset: 0; width: 0%; background: linear-gradient(90deg, #22C55E, #7CFC00); }
    .tp-bar-fill-market { position: absolute; inset: 0; width: 0%; background: linear-gradient(90deg, #3b82f6, #6366f1); opacity: 0.85; }
    .tp-edge-pos { color: #22C55E; font-weight: 600; font-size: 1rem; }
    .tp-edge-neg { color: #EF4444; font-weight: 600; font-size: 1rem; }
    .tp-edge-neutral { color: #9CA3AF; font-weight: 500; font-size: 0.95rem; }
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #7CFC00 !important;
        color: #000000 !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    .deuce-terminal-title {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #F9FAFB !important;
        margin-bottom: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _bundle_metadata_for_ui(model_path: str) -> Optional[Dict[str, Any]]:
    if not model_path or not Path(model_path).exists():
        return None
    try:
        obj, feat_names, _ = load_model(model_path)
        if not isinstance(obj, dict) or obj.get("version") not in (2, 3):
            return {
                "format": "legacy",
                "path": model_path,
                "n_features": len(feat_names) if feat_names else None,
            }
        cal = obj.get("calibration") if isinstance(obj.get("calibration"), dict) else {}
        return {
            "format": "bundle_v%s" % obj.get("version"),
            "model_tier": obj.get("model_tier") or obj.get("model_type"),
            "feature_set_name": obj.get("feature_set_name"),
            "split_strategy": obj.get("split_strategy"),
            "calibration_method": cal.get("method") or obj.get("calibration_requested"),
            "val_log_loss_calibrated": cal.get("val_log_loss_calibrated"),
            "train_date_window": obj.get("train_date_window"),
            "validation_date_window": obj.get("validation_date_window"),
            "test_date_window": obj.get("test_date_window"),
            "n_samples_total": obj.get("n_samples_total"),
            "n_train": obj.get("n_train"),
            "n_val": obj.get("n_val"),
            "n_test_holdout": obj.get("n_test_holdout"),
            "created_at": obj.get("created_at"),
            "code_version": obj.get("code_version"),
            "n_features": len(obj.get("feature_names") or feat_names or []),
            "train_files": obj.get("train_files"),
        }
    except Exception as e:
        return {"error": str(e), "path": model_path}


def render_header():
    st.markdown(
        '<div class="deuce-terminal-title">DEUCE · Tennis market terminal</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Pre-match pricing · no-vig fair · edge vs market · Core / Enhanced routing · calibrated research bundles"
    )
    core_path = get_default_core_path()
    enh_path = get_default_enhanced_path()
    legacy_path = get_default_model_path()
    core_ok = bool(core_path and Path(core_path).exists())
    enh_ok = bool(enh_path and Path(enh_path).exists())
    legacy_ok = bool(legacy_path and Path(legacy_path).exists())
    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("Core bundle", "OK" if core_ok else "Missing")
    with h2:
        st.metric("Enhanced", "OK" if enh_ok else "—")
    with h3:
        st.metric("Legacy", "OK" if legacy_ok else "—")
    st.divider()
    return core_ok, enh_ok, legacy_ok, core_path, enh_path, legacy_path


def _collect_atp_datasets() -> Dict[str, Any]:
    atp_datasets: Dict[str, Any] = {}
    try:
        saved_db = load_db(DEFAULT_DB_PATH)
        if saved_db:
            atp_datasets["player_db"] = saved_db
    except Exception:
        pass
    if st.session_state.atp_data is not None:
        atp_datasets["atp_rankings"] = st.session_state.atp_data[0]
    if st.session_state.serve_data is not None:
        atp_datasets["serve_leaderboard"] = st.session_state.serve_data[0]
    if st.session_state.return_data is not None:
        atp_datasets["return_leaderboard"] = st.session_state.return_data[0]
    if st.session_state.pressure_data is not None:
        atp_datasets["pressure_leaderboard"] = st.session_state.pressure_data[0]
    if st.session_state.winloss_data is not None:
        atp_datasets["winloss_index"] = st.session_state.winloss_data[0]
    if st.session_state.tdi_data is not None:
        atp_datasets["tdi_leaderboard"] = st.session_state.tdi_data[0]
    return atp_datasets


for _k in (
    "matchups_data",
    "atp_data",
    "serve_data",
    "return_data",
    "pressure_data",
    "winloss_data",
    "tdi_data",
    "wf_json_path",
    "wf_oos_csv_path",
    "show_predictions",
):
    if _k not in st.session_state:
        st.session_state[_k] = None
if st.session_state.wf_json_path is None:
    st.session_state.wf_json_path = str(Path.cwd() / "artifacts" / "wf_report.json")
st.session_state.show_predictions = True

_core_ok, _enh_ok, _legacy_ok, _core_path, _enh_path, _legacy_path = render_header()
_meta_core = _bundle_metadata_for_ui(_core_path) if _core_ok and _core_path else None
_bundle_line = build_bundle_summary_line(_core_ok, _core_path, _meta_core)

# --- Shared player DB / catalog for PvP ---
_player_db = load_db(DEFAULT_DB_PATH)
_player_names = sorted([r.get("player") or "" for r in _player_db if r.get("player")], key=str.lower)
try:
    from deuce.tournament_catalog import DEFAULT_CATALOG_PATH, load_catalog

    _catalog_path = Path(DEFAULT_CATALOG_PATH)
    if not _catalog_path.is_absolute():
        _catalog_path = Path.cwd() / _catalog_path
    _catalog = load_catalog(str(_catalog_path)) if _catalog_path.exists() else []
    _tournament_names = sorted([e.get("name") or "" for e in _catalog if e.get("name")], key=str.lower)
except Exception:
    _catalog = []
    _tournament_names = []

with st.sidebar:
    st.header("Terminal filters")
    st.selectbox(
        "Sort by",
        ["edge", "confidence", "completeness"],
        key="term_sort",
    )
    st.slider("Min |edge| filter", 0.0, 0.25, 0.0, 0.005, key="term_min_edge")
    st.slider("Min completeness filter", 0.0, 1.0, 0.0, 0.05, key="term_min_comp")
    st.selectbox(
        "Route",
        ["all", "core", "enhanced", "legacy", "fallback"],
        key="term_route",
    )
    st.selectbox(
        "Tour (event text)",
        ["all", "atp", "wta"],
        key="term_tour",
        help="Heuristic: WTA if event contains 'WTA'; ATP excludes those rows.",
    )
    st.selectbox(
        "Market quality",
        ["all", "exclude_thin", "good_only"],
        key="term_mq",
    )
    st.text_input(
        "Search players, matches, or results",
        value="",
        key="term_search",
        help="Matches player names, matchup/event text, tournament names, round labels, and live-score/result text.",
    )
    st.slider("Bet narrative: min |edge|", 0.0, 0.15, 0.03, 0.005, key="term_bet_min_edge")
    st.slider("Bet narrative: min completeness", 0.0, 1.0, 0.55, 0.05, key="term_bet_min_comp")
    _wf_in = st.text_input(
        "Walk-forward / research JSON path",
        value=st.session_state.wf_json_path or "",
        key="wf_json_sidebar",
        help="Precomputed walk-forward or research export (JSON).",
    )
    if _wf_in:
        st.session_state.wf_json_path = _wf_in.strip()
    _oos_in = st.text_input(
        "Optional OOS CSV (equity / scatter)",
        value=st.session_state.wf_oos_csv_path or "",
        key="wf_oos_csv_sidebar",
        help="From walk-forward `--out-oos-csv` for bankroll curve or model vs market columns.",
    )
    if _oos_in is not None:
        st.session_state.wf_oos_csv_path = _oos_in.strip()

    with st.expander("Expected research artifacts (JSON / OOS CSV)", expanded=False):
        st.markdown("**Walk-forward JSON** (from `aggregate_report_to_json`)")
        st.markdown(WALK_FORWARD_JSON_FIELDS_HELP)
        st.markdown("**OOS CSV**")
        st.markdown(OOS_CSV_COLUMNS_HELP)

    st.divider()
    st.header("Kalshi matchups")
    player = st.text_input(
        "Player name (Kalshi)",
        value="",
        placeholder="e.g. zverev or leave empty for all",
        help="Case-insensitive. Only matchups involving this player, or leave empty for all.",
    )
    days = st.slider("Matchups in next N days", 1, 14, 3, key="days")
    max_results = st.slider("Max matchups shown", 10, 150, 80, 10, key="max_matchups")
    load_matchups = st.button("Refresh matchups", type="primary", key="load_matchups")
    st.caption("Model predictions are always on for matchup cards and rankings.")
    st.divider()
    st.caption(
        "Research datasets, walk-forward artifacts, and ATP leaderboards now live in the "
        "**Research & data** tab to keep the trading desk focused."
    )

st.sidebar.caption(
    "Data: Kalshi API, ATP Tour, TennisMyLife (training). "
    "Core/Enhanced/legacy bundles from model_predict default paths."
)

# --- Auto-load matchups on first visit, then allow manual refresh ---
_requested_player = player.strip()
_requested_ctx = (_requested_player, days, max_results)
_current_ctx = None
if st.session_state.matchups_data is not None:
    _stored_matchups, _stored_player, _stored_days, _stored_max_results = st.session_state.matchups_data
    _current_ctx = (_stored_player, _stored_days, _stored_max_results)

if load_matchups or st.session_state.matchups_data is None or _current_ctx != _requested_ctx:
    with st.spinner("Loading Kalshi tennis matchups..."):
        matchups, fetch_msg = _cached_kalshi_matchups(
            _requested_player,
            days,
            max_results,
        )
    st.session_state.matchups_data = (matchups, _requested_player, days, max_results, fetch_msg)

def _terminal_matchups() -> Tuple[Optional[List[Dict[str, Any]]], bool, Optional[Tuple[str, int]]]:
    if st.session_state.matchups_data is None:
        return None, True, None
    matchups, used_player, used_days, _used_max_results, _fetch_msg = st.session_state.matchups_data
    used_fallback = True
    matchups, used_fallback = enrich_matchups_with_predictions(matchups, _collect_atp_datasets())
    from api import parse_iso8601_utc
    from deuce.live_scores import overlay_live_scores

    date_keys = []
    for m in matchups:
        dt = parse_iso8601_utc(m.get("scheduled_start") or m.get("when"))
        if dt is not None:
            date_keys.append(dt.astimezone().strftime("%Y%m%d"))
    scoreboard = _cached_live_scoreboard(tuple(sorted(set(date_keys)))) if date_keys else []
    matchups = overlay_live_scores(matchups, scoreboard)
    return matchups, used_fallback, (used_player, used_days)


def _filter_kw(bet_only: bool) -> Dict[str, Any]:
    return {
        "search": st.session_state.get("term_search", ""),
        "min_completeness": float(st.session_state.get("term_min_comp", 0.0)),
        "route_filter": st.session_state.get("term_route", "all"),
        "bet_only": bet_only,
        "sort_by": st.session_state.get("term_sort", "edge"),
        "tour_filter": st.session_state.get("term_tour", "all"),
        "market_quality_filter": st.session_state.get("term_mq", "all"),
    }


def _render_markets_tab(*, bet_only: bool) -> None:
    matchups, used_fallback, ctx = _terminal_matchups()
    if matchups is None:
        st.info("Click **Load matchups** in the sidebar to fetch Kalshi markets.")
        return
    used_player, used_days = ctx if ctx else ("", days)
    if used_player:
        st.caption(f"Matchups involving **{used_player}** (next {used_days} days).")
    else:
        st.caption(f"All upcoming singles match-winner markets (next {used_days} days).")
    if not matchups:
        st.warning("No matchups found. Try a different player or increase days.")
        return
    show_pred = True
    if used_fallback:
        st.caption("Using **fallback heuristic** (no trained model or install model + ATP data for model predictions).")
    else:
        st.caption("Using **trained model** — edge vs **fair no-vig** Kalshi price.")

    st.markdown(f"**Active filters:** {format_active_filters_summary(dict(st.session_state))}")

    stats = summarize_matchups_kpis(matchups, show_predictions=True)
    render_kpi_strip(stats, _bundle_line)
    _wf_live = (st.session_state.get("wf_json_path") or "").strip()
    if _wf_live:
        _p_wf = Path(_wf_live)
        if _p_wf.is_file():
            from deuce.presentation.research_credibility import format_credibility_caption

            _wf_data = _cached_walk_forward_json(_wf_live, _p_wf.stat().st_mtime)
            if _wf_data:
                st.caption(format_credibility_caption(_wf_data))
    render_live_market_charts(matchups, show_predictions=True)

    kw = _filter_kw(bet_only)
    me = float(st.session_state.get("term_min_edge", 0.0))
    filtered = filter_matchups(matchups, min_edge=me, **kw)
    soft = filter_matchups(matchups, min_edge=0.0, **kw)
    st.caption(
        filter_diagnostics(
            n_total=len(matchups),
            n_after=len(filtered),
            n_if_edge_floor_removed=len(soft),
        )
    )
    min_edge_card = float(st.session_state.get("term_bet_min_edge", 0.03))
    min_comp_card = float(st.session_state.get("term_bet_min_comp", 0.55))
    compact_live = st.toggle("Compact cards", value=False, key="live_compact_cards")
    for m in filtered:
        render_rich_matchup_card(
            m,
            show_predictions=True,
            min_edge=min_edge_card,
            min_completeness=min_comp_card,
            compact=compact_live,
            live_auto_refresh=True,
            live_refresh_seconds=LIVE_SCORE_REFRESH_SECONDS,
        )


def _render_bet_tab() -> None:
    matchups, used_fallback, ctx = _terminal_matchups()
    if matchups is None:
        st.info("Click **Load matchups** in the sidebar to fetch Kalshi markets.")
        return
    st.markdown(f"**Active filters:** {format_active_filters_summary(dict(st.session_state))}")
    kw = _filter_kw(bet_only=True)
    me = float(st.session_state.get("term_min_edge", 0.0))
    filtered = filter_matchups(matchups, min_edge=me, **kw)
    soft = filter_matchups(matchups, min_edge=0.0, **kw)
    st.caption(
        filter_diagnostics(
            n_total=len(matchups),
            n_after=len(filtered),
            n_if_edge_floor_removed=len(soft),
        )
    )
    min_edge_card = float(st.session_state.get("term_bet_min_edge", 0.03))
    min_comp_card = float(st.session_state.get("term_bet_min_comp", 0.55))
    render_bet_candidates_board(
        filtered,
        matchups,
        show_predictions=True,
        min_edge_card=min_edge_card,
        min_comp_card=min_comp_card,
        live_auto_refresh=True,
        live_refresh_seconds=LIVE_SCORE_REFRESH_SECONDS,
    )


def _render_trading_desk() -> None:
    matchups, used_fallback, ctx = _terminal_matchups()
    st.caption(
        "Primary workflow: load Kalshi matchups, apply trading filters, then review the action board before drilling into the market tape."
    )
    if matchups is None:
        st.info("Click **Load matchups** in the sidebar to start the trading desk.")
        return
    used_player, used_days = ctx if ctx else ("", st.session_state.get("days", 3))
    show_pred = True
    stats = summarize_matchups_kpis(matchups, show_predictions=True)
    render_kpi_strip(stats, _bundle_line)

    left, right = st.columns([1.7, 1.05])
    with left:
        st.markdown(
            """
            <div class="tp-card">
              <div class="tp-card-header">Session focus</div>
              <div style="font-size:1.05rem;font-weight:600;margin:0.35rem 0 0.25rem 0;">Trade the highest-conviction board first</div>
              <div style="font-size:0.9rem;color:#CBD5E1;">
                Keep the main desk centered on priced matchups, edge, completeness, and route quality.
                Deep ATP leaderboards and walk-forward diagnostics stay in Research &amp; data.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if used_player:
            st.caption(f"Universe: **{used_player}** over the next **{used_days}** day(s).")
        else:
            st.caption(f"Universe: all upcoming singles markets over the next **{used_days}** day(s).")
        if used_fallback:
            st.caption("Pricing mode: **fallback heuristic**. Load model-ready datasets for trained routing.")
        else:
            st.caption("Pricing mode: **trained model vs no-vig fair market**.")
    with right:
        st.markdown(
            f"""
            <div class="tp-card">
              <div class="tp-card-header">Desk checklist</div>
              <div style="display:grid;gap:0.45rem;font-size:0.88rem;color:#E5E7EB;">
                <div>1. Load matchups from the sidebar.</div>
                <div>2. Review the always-on model edge and fair price.</div>
                <div>3. Tighten route, quality, and completeness filters.</div>
                <div>4. Use Research &amp; data only when you need deeper context.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    desk_action, desk_tape = st.tabs(["Action board", "Market tape"])
    with desk_action:
        st.caption("Direct trading queue: candidates, watchlist names, and filter diagnostics.")
        _render_bet_tab()
    with desk_tape:
        st.caption("Full market tape: every filtered matchup card, plus distribution views when predictions are on.")
        _render_markets_tab(bet_only=False)


def _render_research_hub() -> None:
    st.caption(
        "Research & data contains walk-forward evidence, ATP context tables, model operations, and player-database updates."
    )
    research_tab, atp_tab, ops_tab = st.tabs(["Walk-forward", "ATP context", "Model ops"])

    with research_tab:
        wf_path = st.text_input(
            "Walk-forward / research JSON path",
            value=st.session_state.wf_json_path or "",
            key="wf_json_main",
            help="Precomputed walk-forward or research export (JSON).",
        )
        st.session_state.wf_json_path = wf_path.strip() if wf_path else ""
        oos_path = st.text_input(
            "Optional OOS CSV path",
            value=st.session_state.wf_oos_csv_path or "",
            key="wf_oos_csv_main",
            help="From walk-forward `--out-oos-csv` for bankroll curve or model vs market columns.",
        )
        st.session_state.wf_oos_csv_path = oos_path.strip() if oos_path else ""
        with st.expander("Expected research artifacts (JSON / OOS CSV)", expanded=False):
            st.markdown("**Walk-forward JSON** (from `aggregate_report_to_json`)")
            st.markdown(WALK_FORWARD_JSON_FIELDS_HELP)
            st.markdown("**OOS CSV**")
            st.markdown(OOS_CSV_COLUMNS_HELP)
        render_research_tab(
            st.session_state.wf_json_path or "",
            oos_csv_path=st.session_state.wf_oos_csv_path or "",
        )

    with atp_tab:
        st.markdown("### ATP context loaders")
        st.caption("Load only the contextual datasets you want on demand. Nothing here is required for basic market browsing.")

        c1, c2 = st.columns(2)
        with c1:
            top_n = st.number_input(
                "Top N ATP rank",
                min_value=10,
                max_value=500,
                value=50,
                step=10,
                help="Show top N players in ATP singles rankings.",
                key="top_n",
            )
        with c2:
            if st.button("Load ATP rankings", key="load_atp"):
                with st.spinner("Loading ATP rankings..."):
                    atp_rows = get_atp_rankings_display(limit=top_n)
                st.session_state.atp_data = (atp_rows, top_n)

        c3, c4, c5 = st.columns(3)
        with c3:
            serve_surface = st.selectbox(
                "Serve surface",
                options=["all", "Clay", "Grass", "Hard"],
                index=0,
                key="serve_surface",
            )
        with c4:
            serve_limit = st.number_input(
                "Serve top N",
                min_value=10,
                max_value=100,
                value=25,
                step=5,
                key="serve_limit",
            )
        with c5:
            if st.button("Load Serve leaderboard", key="load_serve"):
                with st.spinner("Loading ATP Serve leaderboard (this may take a moment)..."):
                    serve_rows = get_atp_serve_leaderboard_display(
                        surface=serve_surface,
                        limit=serve_limit,
                    )
                st.session_state.serve_data = (serve_rows, serve_surface, serve_limit)

        c6, c7, c8 = st.columns(3)
        with c6:
            return_surface = st.selectbox(
                "Return surface",
                options=["all", "Clay", "Grass", "Hard"],
                index=0,
                key="return_surface",
            )
        with c7:
            return_limit = st.number_input("Return top N", min_value=10, max_value=100, value=25, step=5, key="return_limit")
        with c8:
            if st.button("Load Return leaderboard", key="load_return"):
                with st.spinner("Loading ATP Return leaderboard (this may take a moment)..."):
                    return_rows = get_atp_return_leaderboard_display(surface=return_surface, limit=return_limit)
                st.session_state.return_data = (return_rows, return_surface, return_limit)

        c9, c10, c11 = st.columns(3)
        with c9:
            pressure_surface = st.selectbox(
                "Pressure surface",
                options=["all", "Clay", "Grass", "Hard"],
                index=0,
                key="pressure_surface",
            )
        with c10:
            pressure_limit = st.number_input(
                "Pressure top N",
                min_value=10,
                max_value=100,
                value=25,
                step=5,
                key="pressure_limit",
            )
        with c11:
            if st.button("Load Under Pressure leaderboard", key="load_pressure"):
                with st.spinner("Loading ATP Under Pressure leaderboard (this may take a moment)..."):
                    pressure_rows = get_atp_pressure_leaderboard_display(surface=pressure_surface, limit=pressure_limit)
                st.session_state.pressure_data = (pressure_rows, pressure_surface, pressure_limit)

        winloss_category_labels = [label for _, label in ATP_WINLOSS_INDEX_CATEGORIES]
        winloss_time_labels = [label for _, label in ATP_WINLOSS_TIME_CATEGORIES]
        c12, c13, c14, c15 = st.columns(4)
        with c12:
            winloss_category_label = st.selectbox(
                "Win/Loss category",
                options=winloss_category_labels,
                index=0,
                key="winloss_category",
            )
        with c13:
            winloss_time_label = st.selectbox(
                "Win/Loss time",
                options=winloss_time_labels,
                index=0,
                key="winloss_time",
            )
        with c14:
            winloss_limit = st.number_input(
                "Win/Loss top N",
                min_value=10,
                max_value=100,
                value=25,
                step=5,
                key="winloss_limit",
            )
        with c15:
            if st.button("Load Win/Loss Index", key="load_winloss"):
                winloss_category_slug = next(
                    slug for slug, label in ATP_WINLOSS_INDEX_CATEGORIES if label == winloss_category_label
                )
                winloss_time_slug = next(
                    slug for slug, label in ATP_WINLOSS_TIME_CATEGORIES if label == winloss_time_label
                )
                with st.spinner("Loading ATP Win/Loss Index (this may take a moment)..."):
                    winloss_rows = get_atp_winloss_index_display(
                        index_category=winloss_category_slug,
                        time_category=winloss_time_slug,
                        limit=winloss_limit,
                    )
                st.session_state.winloss_data = (
                    winloss_rows,
                    winloss_category_label,
                    winloss_time_label,
                    winloss_limit,
                )

        tdi_time_labels = [label for _, label in ATP_TDI_TIME_FRAMES]
        c16, c17, c18, c19 = st.columns(4)
        with c16:
            tdi_time_label = st.selectbox(
                "TDI time",
                options=tdi_time_labels,
                index=0,
                key="tdi_time",
            )
        with c17:
            tdi_surface = st.selectbox(
                "TDI surface",
                options=["all", "Clay", "Grass", "Hard"],
                index=0,
                key="tdi_surface",
            )
        with c18:
            tdi_limit = st.number_input(
                "TDI top N",
                min_value=10,
                max_value=100,
                value=25,
                step=5,
                key="tdi_limit",
            )
        with c19:
            if st.button("Load TDI leaderboard", key="load_tdi"):
                tdi_time_slug = next(slug for slug, label in ATP_TDI_TIME_FRAMES if label == tdi_time_label)
                with st.spinner("Loading ATP TDI Insights Leaderboard (this may take a moment)..."):
                    tdi_rows = get_atp_tdi_leaderboard_display(
                        time_frame=tdi_time_slug,
                        surface=tdi_surface,
                        limit=tdi_limit,
                    )
                st.session_state.tdi_data = (tdi_rows, tdi_time_label, tdi_surface, tdi_limit)

        st.divider()
        save_to_db = st.button("Save loaded ATP data to player database", key="save_to_db")
        if save_to_db:
            atp_rankings = st.session_state.atp_data[0] if st.session_state.atp_data else None
            serve_rows = st.session_state.serve_data[0] if st.session_state.serve_data else None
            return_rows = st.session_state.return_data[0] if st.session_state.return_data else None
            pressure_rows = st.session_state.pressure_data[0] if st.session_state.pressure_data else None
            winloss_rows = st.session_state.winloss_data[0] if st.session_state.winloss_data else None
            tdi_rows = st.session_state.tdi_data[0] if st.session_state.tdi_data else None
            if atp_rankings or serve_rows or return_rows or pressure_rows or winloss_rows or tdi_rows:
                n = update_db_from_atp_datasets(
                    DEFAULT_DB_PATH,
                    atp_rankings=atp_rankings,
                    serve_leaderboard=serve_rows,
                    return_leaderboard=return_rows,
                    pressure_leaderboard=pressure_rows,
                    winloss_index=winloss_rows,
                    tdi_leaderboard=tdi_rows,
                )
                st.success(f"Saved to **{DEFAULT_DB_PATH}**. Total players: **{n}**.")
            else:
                st.warning("Load at least one ATP dataset first, then save.")

        st.divider()
        _render_atp_sections()

    with ops_tab:
        render_model_operations_panel(_core_ok, _core_path, _meta_core)


def _render_atp_sections() -> None:
    st.subheader("ATP singles rankings")
    if st.session_state.atp_data is not None:
        atp_rows, used_top_n = st.session_state.atp_data
        st.caption(f"Top {used_top_n} (click **Load ATP rankings** to refresh).")
        if not atp_rows:
            st.warning("Could not load ATP rankings.")
        else:
            df = pd.DataFrame(atp_rows)
            df = df.rename(
                columns={
                    "rank": "Rank",
                    "player": "Player",
                    "movement": "Movement",
                    "age": "Age",
                    "official points": "Official Points",
                    "pts chg": "Pts Chg",
                    "tourn played": "Tourn Played",
                }
            )
            st.dataframe(
                df[["Rank", "Player", "Movement", "Age", "Official Points", "Pts Chg", "Tourn Played"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Click **Load ATP rankings** in the sidebar to fetch (no Kalshi request).")

    st.divider()
    st.subheader("ATP Serve leaderboard (52 weeks)")
    if st.session_state.serve_data is not None:
        serve_rows, used_surface, used_limit = st.session_state.serve_data
        st.caption(f"Surface: **{used_surface}** | Top {used_limit} (click **Load Serve leaderboard** to refresh).")
        if not serve_rows:
            st.warning("Could not load Serve leaderboard. Install Playwright: pip install playwright && playwright install chromium")
        else:
            df_serve = pd.DataFrame(serve_rows)
            df_serve = df_serve.rename(
                columns={
                    "rank": "Rank",
                    "player": "Player",
                    "serve_rating": "Serve Rating",
                    "pct_1st_serve": "% 1st Serve",
                    "pct_1st_serve_pts_won": "% 1st Serve Pts Won",
                    "pct_2nd_serve_pts_won": "% 2nd Serve Pts Won",
                    "pct_service_games_won": "% Service Games Won",
                    "avg_aces_per_match": "Avg Aces/Match",
                    "avg_double_faults_per_match": "Avg Dbl Faults/Match",
                }
            )
            display_cols = [
                "Rank",
                "Player",
                "Serve Rating",
                "% 1st Serve",
                "% 1st Serve Pts Won",
                "% 2nd Serve Pts Won",
                "% Service Games Won",
                "Avg Aces/Match",
                "Avg Dbl Faults/Match",
            ]
            st.dataframe(
                df_serve[[c for c in display_cols if c in df_serve.columns]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Click **Load Serve leaderboard** in the sidebar to fetch (uses Playwright).")

    st.divider()
    st.subheader("ATP Return leaderboard (52 weeks)")
    if st.session_state.return_data is not None:
        return_rows, used_surface, used_limit = st.session_state.return_data
        st.caption(f"Surface: **{used_surface}** | Top {used_limit} (click **Load Return leaderboard** to refresh).")
        if not return_rows:
            st.warning("Could not load Return leaderboard.")
        else:
            df_return = pd.DataFrame(return_rows)
            df_return = df_return.rename(
                columns={
                    "rank": "Rank",
                    "player": "Player",
                    "return_rating": "Return Rating",
                    "pct_1st_serve_return_pts_won": "% 1st Serve Return Pts Won",
                    "pct_2nd_serve_return_pts_won": "% 2nd Serve Return Pts Won",
                    "pct_return_games_won": "% Return Games Won",
                    "pct_break_pts_converted": "% Break Pts Converted",
                }
            )
            display_cols = [
                "Rank",
                "Player",
                "Return Rating",
                "% 1st Serve Return Pts Won",
                "% 2nd Serve Return Pts Won",
                "% Return Games Won",
                "% Break Pts Converted",
            ]
            st.dataframe(
                df_return[[c for c in display_cols if c in df_return.columns]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Click **Load Return leaderboard** in the sidebar to fetch.")

    st.divider()
    st.subheader("ATP Under Pressure leaderboard (52 weeks)")
    if st.session_state.pressure_data is not None:
        pressure_rows, used_surface, used_limit = st.session_state.pressure_data
        st.caption(f"Surface: **{used_surface}** | Top {used_limit} (click **Load Under Pressure leaderboard** to refresh).")
        if not pressure_rows:
            st.warning("Could not load Under Pressure leaderboard.")
        else:
            df_pressure = pd.DataFrame(pressure_rows)
            df_pressure = df_pressure.rename(
                columns={
                    "rank": "Rank",
                    "player": "Player",
                    "pressure_rating": "Under Pressure Rating",
                    "pct_break_pts_converted": "% Break Pts Converted",
                    "pct_break_pts_saved": "% Break Pts Saved",
                    "pct_tiebreaks_won": "% Tie Breaks Won",
                    "pct_deciding_sets_won": "% Deciding Sets Won",
                }
            )
            display_cols = [
                "Rank",
                "Player",
                "Under Pressure Rating",
                "% Break Pts Converted",
                "% Break Pts Saved",
                "% Tie Breaks Won",
                "% Deciding Sets Won",
            ]
            st.dataframe(
                df_pressure[[c for c in display_cols if c in df_pressure.columns]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Click **Load Under Pressure leaderboard** in the sidebar to fetch.")

    st.divider()
    st.subheader("ATP Win/Loss Index")
    if st.session_state.winloss_data is not None:
        winloss_rows, used_category, used_time, used_limit = st.session_state.winloss_data
        st.caption(f"**{used_category}** — {used_time} | Top {used_limit} (click **Load Win/Loss Index** to refresh).")
        if not winloss_rows:
            st.warning("Could not load Win/Loss Index.")
        else:
            df_winloss = pd.DataFrame(winloss_rows)
            df_winloss = df_winloss.rename(
                columns={
                    "rank": "Rank",
                    "player": "Player",
                    "index": "Index",
                    "titles": "Titles",
                    "win_loss": "Win/Loss",
                }
            )
            display_cols = ["Rank", "Player", "Index", "Titles", "Win/Loss"]
            st.dataframe(
                df_winloss[[c for c in display_cols if c in df_winloss.columns]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Click **Load Win/Loss Index** in the sidebar to fetch (choose category and time).")

    st.divider()
    st.subheader("ATP TDI Insights Leaderboard")
    if st.session_state.tdi_data is not None:
        tdi_rows, used_time, used_surface, used_limit = st.session_state.tdi_data
        st.caption(f"**{used_time}** | Surface: **{used_surface}** | Top {used_limit} (click **Load TDI leaderboard** to refresh).")
        if not tdi_rows:
            st.warning("Could not load TDI leaderboard.")
        else:
            df_tdi = pd.DataFrame(tdi_rows)
            df_tdi = df_tdi.rename(
                columns={
                    "rank": "Rank",
                    "player": "Player",
                    "performance_rating": "Perf. Rating",
                    "conversion_pct": "Convs.%",
                    "steal_pct": "Steal%",
                    "serve": "Serve",
                    "return_shot": "Return",
                    "forehand": "Forehand",
                    "backhand": "Backhand",
                }
            )
            display_cols = [
                "Rank",
                "Player",
                "Perf. Rating",
                "Convs.%",
                "Steal%",
                "Serve",
                "Return",
                "Forehand",
                "Backhand",
            ]
            st.dataframe(
                df_tdi[[c for c in display_cols if c in df_tdi.columns]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.caption("Click **Load TDI leaderboard** in the sidebar to fetch (Performance, Conversion, Steal, Shot Quality).")


tab_desk, tab_pvp, tab_research, tab_dev = st.tabs(
    [
        "Trading desk",
        "Match lab",
        "Research & data",
        "Developer",
    ]
)

with tab_desk:
    st.markdown("### Trading desk")
    _render_trading_desk()

with tab_pvp:
    st.markdown("### Match lab")
    render_pvp_lab(
        player_names=_player_names,
        player_db=_player_db,
        catalog=_catalog,
        tournament_names=_tournament_names,
        core_ok=_core_ok,
        enh_ok=_enh_ok,
        legacy_ok=_legacy_ok,
        core_path=_core_path,
        enh_path=_enh_path,
        legacy_path=_legacy_path,
    )

with tab_research:
    st.markdown("### Research & data")
    _render_research_hub()

_matchups_dev, _, _ = _terminal_matchups()
with tab_dev:
    st.markdown("### Developer / raw data")
    render_developer_tab(
        _matchups_dev,
        wf_path=st.session_state.wf_json_path or "",
        core_path=_core_path,
        enh_path=_enh_path,
        legacy_path=_legacy_path,
        show_predictions=True,
    )
