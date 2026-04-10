"""Head-to-head lab: deep inspection for one synthetic matchup."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from deuce.presentation.bet_signal import decision_record, decision_strip_html
from model_predict import load_model

_BUCKET_LABELS = {
    "base_strength": "Base strength (rank/points)",
    "rank_points": "Rank / points",
    "tennis_skill": "Tennis skill (serve/return/pressure)",
    "context": "Context (surface, round, draw)",
    "match_level": "Match-level stats",
    "interactions": "Interactions",
    "recent_form": "Recent form",
    "draw_status": "Draw status (seed)",
    "event_level": "Event level",
    "entry_path": "Entry path",
    "event_identity": "Event identity",
    "score_context": "Score context",
    "nationality_home": "Nationality / home",
    "match_progression": "Match progression",
    "uncertainty_penalty": "Uncertainty (missing-data penalty)",
}


def render_route_badge(route: str) -> None:
    r = (route or "—").lower()
    badge = {
        "core": ("CORE", "tp-badge-med"),
        "enhanced": ("ENHANCED", "tp-badge-high"),
        "legacy": ("LEGACY", "tp-badge-low"),
        "fallback": ("FALLBACK", "tp-badge-low"),
    }.get(r, (route.upper()[:12], "tp-badge-low"))
    st.markdown(
        f'<span class="tp-badge {badge[1]}">{badge[0]}</span>',
        unsafe_allow_html=True,
    )


def render_pvp_lab(
    *,
    player_names: List[str],
    player_db: List[Dict[str, Any]],
    catalog: List[Dict[str, Any]],
    tournament_names: List[str],
    core_ok: bool,
    enh_ok: bool,
    legacy_ok: bool,
    core_path: Optional[str],
    enh_path: Optional[str],
    legacy_path: Optional[str],
) -> None:
    if not player_names:
        st.caption("No players in DB. Load ATP data and **Save to player database** to populate.")
        return

    col_a, col_b, col_t = st.columns(3)
    with col_a:
        player_a_name = st.selectbox("Player A", options=player_names, key="pvp_player_a")
    with col_b:
        player_b_name = st.selectbox("Player B", options=player_names, key="pvp_player_b")
    with col_t:
        tournament_options = ["(no tournament)"] + tournament_names
        tournament_choice = st.selectbox("Tournament (optional)", options=tournament_options, key="pvp_tournament")
    tournament_name = None if (not tournament_choice or tournament_choice == "(no tournament)") else tournament_choice

    if not (player_a_name and player_b_name and player_a_name != player_b_name):
        if player_a_name == player_b_name and player_a_name:
            st.caption("Choose two different players.")
        return

    try:
        from deuce.features.coverage_features import has_atp_leaderboard_coverage
        from model_features import build_player_stats_from_datasets
        from model_predict import predict_match_probability
        from deuce.player_db import get_player_stats

        _stats_a = build_player_stats_from_datasets(player_a_name, player_db=player_db)
        _stats_b = build_player_stats_from_datasets(player_b_name, player_db=player_db)

        _rec_a = get_player_stats(player_db, player_a_name)
        _rec_b = get_player_stats(player_db, player_b_name)
        _def = lambda r, k: (r or {}).get(k) if r else None

        with st.expander("Compare player stats (from DB)", expanded=True):
            _cmp = pd.DataFrame(
                {
                    "Stat": [
                        "Rank",
                        "Points",
                        "Serve rating",
                        "Return rating",
                        "Pressure rating",
                        "Win/loss index",
                        "TDI perf",
                    ],
                    player_a_name: [
                        _def(_rec_a, "rank"),
                        _def(_rec_a, "points"),
                        _def(_rec_a, "serve_rating"),
                        _def(_rec_a, "return_rating"),
                        _def(_rec_a, "pressure_rating"),
                        _def(_rec_a, "winloss_index"),
                        _def(_rec_a, "tdi_perf"),
                    ],
                    player_b_name: [
                        _def(_rec_b, "rank"),
                        _def(_rec_b, "points"),
                        _def(_rec_b, "serve_rating"),
                        _def(_rec_b, "return_rating"),
                        _def(_rec_b, "pressure_rating"),
                        _def(_rec_b, "winloss_index"),
                        _def(_rec_b, "tdi_perf"),
                    ],
                }
            )
            st.dataframe(_cmp, use_container_width=True, hide_index=True)

        def _has(v) -> str:
            return "Yes" if v is not None and v != "" else "No"

        with st.expander("Data coverage (model inputs)", expanded=False):
            st.caption("Feature groups available per player (v3 uses seed, entry, nationality when present).")
            _coverage = pd.DataFrame(
                {
                    "Feature group": [
                        "Rank / points",
                        "Serve / return / pressure",
                        "Win/loss index",
                        "TDI",
                        "Age / height / hand",
                        "Recent form",
                        "Seed",
                        "Nationality",
                    ],
                    player_a_name: [
                        _has(_def(_rec_a, "rank")) or _has(_def(_rec_a, "points")),
                        _has(_def(_rec_a, "serve_rating"))
                        or _has(_def(_rec_a, "return_rating"))
                        or _has(_def(_rec_a, "pressure_rating")),
                        _has(_def(_rec_a, "winloss_index")),
                        _has(_def(_rec_a, "tdi_perf")),
                        _has(_def(_rec_a, "age")) or _has(_def(_rec_a, "height")) or _has(_def(_rec_a, "hand")),
                        _has(getattr(_stats_a, "last5_win_pct", None)) or _has(getattr(_stats_a, "recent_hold_pct", None)),
                        _has(_def(_rec_a, "seed")),
                        _has(_def(_rec_a, "ioc")) or _has(_def(_rec_a, "nationality")),
                    ],
                    player_b_name: [
                        _has(_def(_rec_b, "rank")) or _has(_def(_rec_b, "points")),
                        _has(_def(_rec_b, "serve_rating"))
                        or _has(_def(_rec_b, "return_rating"))
                        or _has(_def(_rec_b, "pressure_rating")),
                        _has(_def(_rec_b, "winloss_index")),
                        _has(_def(_rec_b, "tdi_perf")),
                        _has(_def(_rec_b, "age")) or _has(_def(_rec_b, "height")) or _has(_def(_rec_b, "hand")),
                        _has(getattr(_stats_b, "last5_win_pct", None)) or _has(getattr(_stats_b, "recent_hold_pct", None)),
                        _has(_def(_rec_b, "seed")),
                        _has(_def(_rec_b, "ioc")) or _has(_def(_rec_b, "nationality")),
                    ],
                }
            )
            st.dataframe(_coverage, use_container_width=True, hide_index=True)

        _model_core = _model_enhanced = _model_legacy = None
        _feat_core = _feat_enhanced = _feat_legacy = None
        if core_ok and core_path:
            _model_core, _feat_core, _ = load_model(core_path)
        if enh_ok and enh_path:
            _model_enhanced, _feat_enhanced, _ = load_model(enh_path)
        if legacy_ok and legacy_path:
            _model_legacy, _feat_legacy, _ = load_model(legacy_path)

        _catalog_for_pred = catalog if tournament_name and catalog else None
        _res = predict_match_probability(
            _stats_a,
            _stats_b,
            model=_model_legacy,
            model_core=_model_core,
            model_enhanced=_model_enhanced,
            tournament_name=tournament_name,
            catalog=_catalog_for_pred,
        )

        r1, r2 = st.columns([3, 1])
        with r1:
            st.markdown("#### Prediction")
        with r2:
            render_route_badge(str(getattr(_res, "model_used", "legacy")))

        pvp_cols = st.columns([1, 1, 1])
        with pvp_cols[0]:
            st.markdown(f"**{player_a_name}**")
            st.markdown(f"<span style='font-size:1.4rem;'>{_res.model_prob_a:.1%}</span>", unsafe_allow_html=True)
        with pvp_cols[1]:
            st.markdown(f"**{player_b_name}**")
            st.markdown(f"<span style='font-size:1.4rem;'>{_res.model_prob_b:.1%}</span>", unsafe_allow_html=True)
        with pvp_cols[2]:
            edge_val = _res.model_prob_a - 0.5
            edge_cls = "tp-edge-pos" if edge_val > 0.02 else "tp-edge-neg" if edge_val < -0.02 else "tp-edge-neutral"
            st.markdown("**Edge vs 50/50**", unsafe_allow_html=True)
            st.markdown(f"<span class='{edge_cls}'>{edge_val:+.1%}</span>", unsafe_allow_html=True)

        bar_html = f"""
        <div class="tp-card" style="margin-top:0.75rem;">
          <div class="tp-bar-row">
            <span>Model A</span>
            <div class="tp-bar-track">
              <div class="tp-bar-fill-model" style="width:{_res.model_prob_a*100:.1f}%;"></div>
            </div>
            <span>{_res.model_prob_a:.0%}</span>
          </div>
          <div class="tp-bar-row">
            <span>50/50</span>
            <div class="tp-bar-track">
              <div class="tp-bar-fill-market" style="width:50%;"></div>
            </div>
            <span>50%</span>
          </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        _lb = getattr(_res, "leaderboard_coverage_score", None)
        _dc = getattr(_res, "data_completeness", None)
        _mi = (
            f"Evidence: **{getattr(_res, 'evidence_quality', 'unknown')}** · Confidence: **{_res.confidence}**"
        )
        if _dc is not None:
            _mi += f" · Completeness: **{_dc:.0%}**"
        if _lb is not None:
            _mi += f" · Leaderboard coverage: **{_lb:.2f}**"
        st.caption(_mi)

        with st.expander("Decision hygiene vs 50/50 synthetic market", expanded=False):
            st.caption("Same structured checks as Live cards, with **fair price fixed at 50%** (lab only).")
            _syn_edge = float(_res.model_prob_a - 0.5)
            _syn_abs = abs(_syn_edge)
            _low_info = str(_res.confidence).lower() in ("low",)
            _rec_lab = decision_record(
                edge=_syn_edge,
                abs_edge=_syn_abs,
                completeness=_dc,
                market_quality="synthetic",
                route=str(getattr(_res, "model_used", "legacy")),
                edge_reliable=not _low_info,
                low_information=_low_info,
                market_valid=True,
                low_liquidity=False,
            )
            st.markdown(decision_strip_html(_rec_lab), unsafe_allow_html=True)
            st.markdown(f"**{_rec_lab.one_line}**")

        try:
            _atp_ok = has_atp_leaderboard_coverage(_stats_a, _stats_b)
            _rank_a = getattr(_stats_a, "rank", None) is not None or getattr(_stats_a, "points", None) is not None
            _rank_b = getattr(_stats_b, "rank", None) is not None or getattr(_stats_b, "points", None) is not None
            _rank_cov = (
                "both"
                if (_rank_a and _rank_b)
                else ("A only" if _rank_a else ("B only" if _rank_b else "missing"))
            )
            st.caption(
                f"Coverage: Rank/points **{_rank_cov}** · ATP leaderboard (serve/return) "
                f"**{'both' if _atp_ok else 'missing or partial'}**"
            )
        except Exception:
            pass

        try:
            from evaluation import evaluate as evaluate_pred

            _used = getattr(_res, "model_used", "legacy")
            _active_model = _model_core if _used == "core" else (_model_enhanced if _used == "enhanced" else _model_legacy)
            _active_names = _feat_core if _used == "core" else (_feat_enhanced if _used == "enhanced" else _feat_legacy)
            _eval = (
                evaluate_pred(
                    _res,
                    market_prob_a=0.5,
                    market_prob_b=0.5,
                    model=_active_model,
                    feature_names=_active_names,
                )
                if (_active_model is not None and _active_names is not None)
                else None
            )
            if _eval:
                with st.expander("Detailed factor breakdown (grouped buckets)", expanded=False):
                    st.caption("Positive = favors Player A. Pre-match: score_context omitted for Core/Enhanced.")
                    _bd = _eval.feature_breakdown
                    if _bd and _used in ("core", "enhanced"):
                        _bd = {k: v for k, v in _bd.items() if k != "score_context"}
                    try:
                        from model_config import CONTRIBUTION_BUCKETS

                        _order = [b for b in CONTRIBUTION_BUCKETS if _bd and b in _bd]
                    except ImportError:
                        _order = list(_bd.keys()) if _bd else []
                    if _bd:
                        _rows = [(_BUCKET_LABELS.get(k, k.replace("_", " ").title()), _bd[k]) for k in _order if k in _bd]
                        if not _rows:
                            _rows = list(_bd.items())
                        st.dataframe(
                            pd.DataFrame(_rows, columns=["Factor", "Contribution"]),
                            use_container_width=True,
                            hide_index=True,
                        )
                    st.markdown(f"**Summary:** {_eval.explanation}")

                if getattr(_res, "grouped_contributions", None):
                    with st.expander("Top grouped contributions (score buckets)", expanded=False):
                        _gc = pd.DataFrame(list(_res.grouped_contributions.items()), columns=["Bucket", "Contribution"])
                        st.dataframe(_gc, use_container_width=True, hide_index=True)
        except Exception:
            pass

        if getattr(_res, "warnings", None):
            for _w in _res.warnings:
                st.warning(_w)

        if tournament_name and catalog:
            _entry = next(
                (e for e in catalog if (e.get("name") or "").strip().lower() == tournament_name.strip().lower()),
                None,
            )
            if _entry:
                with st.expander("Tournament context (catalog)", expanded=False):
                    st.caption("Training weight, surface, climate features when present.")
                    _cat_df = pd.DataFrame(
                        [
                            ("Name", _entry.get("name") or ""),
                            ("Tier", _entry.get("tier") or ""),
                            ("Weight (training)", _entry.get("weight")),
                            ("Surface", _entry.get("surface") or ""),
                            ("Month", _entry.get("month")),
                            ("Temperature (avg °F)", _entry.get("temperature_avg")),
                            ("Humidity (avg %)", _entry.get("humidity_avg")),
                            ("Location", _entry.get("location") or ""),
                            ("Date start", _entry.get("date_start") or ""),
                            ("Date end", _entry.get("date_end") or ""),
                        ],
                        columns=["Field", "Value"],
                    )
                    st.dataframe(_cat_df, use_container_width=True, hide_index=True)

        _di = getattr(_res, "diagnostics", None)
        if _di:
            with st.expander("Developer debug (raw / scaled / per-feature)", expanded=False):
                if "bundle_version" in _di or "feature_count_in_bundle" in _di:
                    st.caption(
                        f"Bundle **v{_di.get('bundle_version', '?')}** · Features in bundle: **{_di.get('feature_count_in_bundle', '?')}**"
                    )
                if _di.get("missing_count") is not None:
                    st.caption(f"Missing flags: **{_di.get('missing_count')}** / {_di.get('total_features', '?')}")
                if _di.get("raw_feature_values"):
                    st.markdown("**Raw engineered features**")
                    _raw = _di["raw_feature_values"]
                    st.dataframe(
                        pd.DataFrame(list(_raw.items()), columns=["Feature", "Value"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                _raw = _di.get("raw_feature_values") or {}
                _mf = [(n, v) for n, v in _raw.items() if n.endswith("_missing") and (v == 1 or v == 1.0)]
                if _mf:
                    st.markdown("**Missing flags**")
                    st.dataframe(pd.DataFrame(_mf, columns=["Feature", "Value"]), use_container_width=True, hide_index=True)
                if _di.get("scaled_feature_values"):
                    st.markdown("**Scaled features**")
                    _sc = _di["scaled_feature_values"]
                    st.dataframe(
                        pd.DataFrame(list(_sc.items()), columns=["Feature", "Scaled"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                if _di.get("per_feature_contribution"):
                    st.markdown("**Per-feature contribution**")
                    _pf = _di["per_feature_contribution"]
                    st.dataframe(
                        pd.DataFrame(list(_pf.items()), columns=["Feature", "Contribution"]),
                        use_container_width=True,
                        hide_index=True,
                    )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
