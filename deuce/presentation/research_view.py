"""Walk-forward / research JSON dashboard (keys match aggregate_report_to_json)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from deuce.presentation.research_format import (
    as_float,
    format_avg_edge,
    format_brier,
    format_calibration_scalar,
    format_hit_rate,
    format_log_loss,
    format_max_drawdown,
    format_roi_or_yield,
    headline_scale_caption,
)
from deuce.presentation.research_io import load_research_json
from deuce.presentation.research_schema import (
    describe_oos_csv_gaps,
    missing_json_keys,
    oos_csv_column_hints,
)


def _cal_table_rows(cal: Dict[str, Any]) -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    for k, v in cal.items():
        if isinstance(v, list):
            rows.append((k, f"{len(v)} values"))
        elif as_float(v) is not None:
            rows.append((k, format_calibration_scalar(v)))
        else:
            rows.append((k, str(v)))
    return rows


def render_research_tab(wf_path: str, *, oos_csv_path: str = "") -> None:
    """Render precomputed walk-forward JSON; optional OOS CSV for equity / scatter."""
    data = load_research_json(wf_path) if (wf_path or "").strip() else None
    if not data:
        st.info(
            "No research file loaded or path invalid. Export with e.g. "
            "`python -m deuce.cli walk-forward-report --csv data/historical.csv --out-json artifacts/wf_report.json` "
            "and set the path in the sidebar."
        )
        return

    miss = missing_json_keys(data)
    if miss:
        st.warning(
            "Walk-forward JSON is missing expected headline keys (sections may be empty or stale): **"
            + "**, **".join(miss)
            + "**."
        )

    st.caption(
        f"Mode **{data.get('mode', '—')}** · **{data.get('n_folds', '—')}** folds · "
        f"**{data.get('n_oos_predictions', '—')}** OOS predictions"
    )

    st.subheader("Headline OOS metrics")
    roi_s, roi_note = format_roi_or_yield(data.get("roi"))
    yld_s, yld_note = format_roi_or_yield(data.get("yield_per_bet"))
    hr_s, hr_note = format_hit_rate(data.get("hit_rate"))
    ae_s, ae_note = format_avg_edge(data.get("avg_edge"))
    dd_s, dd_note = format_max_drawdown(data.get("max_drawdown"))
    scale_notes = [n for n in (roi_note, yld_note, hr_note, ae_note, dd_note) if n]
    cap = headline_scale_caption(scale_notes)
    if cap:
        st.caption(cap)

    r1 = st.columns(4)
    r1[0].metric("Log loss", format_log_loss(data.get("log_loss")))
    r1[1].metric("Brier", format_brier(data.get("brier")))
    r1[2].metric("ROI", roi_s)
    r1[3].metric("Yield / bet", yld_s)
    r2 = st.columns(4)
    r2[0].metric("Max drawdown", dd_s)
    r2[1].metric("Hit rate", hr_s)
    r2[2].metric("Avg edge", ae_s)
    nb = (data.get("bankroll") or {}).get("n_bets") if isinstance(data.get("bankroll"), dict) else None
    r2[3].metric("n bets (bankroll)", str(int(nb)) if nb is not None else "—")

    bank = data.get("bankroll") or {}
    if isinstance(bank, dict) and bank.get("n_bets", 0):
        broi, bn1 = format_roi_or_yield(bank.get("roi"))
        byld, bn2 = format_roi_or_yield(bank.get("yield_per_bet"))
        bdd, bn3 = format_max_drawdown(bank.get("max_drawdown"))
        bnotes = headline_scale_caption([n for n in (bn1, bn2, bn3) if n])
        st.caption(
            f"Bankroll summary: ROI **{broi}** · yield **{byld}** · max DD **{bdd}**"
            + (f" · {bnotes}" if bnotes else "")
        )

    cal = data.get("calibration_summary") or {}
    st.subheader("Calibration")
    if isinstance(cal, dict) and cal:
        st.dataframe(
            pd.DataFrame(_cal_table_rows(cal), columns=["Metric", "Value"]),
            use_container_width=True,
            hide_index=True,
        )
        pf = cal.get("per_fold_val_ll_calibrated")
        if isinstance(pf, list) and len(pf) >= 2:
            st.caption("Validation log loss (calibrated) by fold index")
            st.line_chart(
                pd.DataFrame(
                    {"fold": range(1, len(pf) + 1), "val_log_loss_cal": [float(x) for x in pf]}
                ).set_index("fold")
            )
    else:
        m = data.get("calibration_mean_val_ll_calibrated")
        if m is not None:
            st.metric("Mean val LL (calibrated)", format_log_loss(m))
        else:
            st.caption("No calibration block in file.")

    folds = data.get("folds") or []
    if folds:
        st.subheader("Fold-by-fold")
        fold_rows: List[Dict[str, Any]] = []
        for f in folds:
            if not isinstance(f, dict):
                continue
            cal_d = f.get("calibration") or {}
            fold_rows.append(
                {
                    "fold": f.get("fold_index"),
                    "n_train": f.get("n_train_fit"),
                    "n_val": f.get("n_val_cal"),
                    "n_oos": f.get("n_oos"),
                    "val_LL_cal": format_log_loss(cal_d.get("val_log_loss_calibrated")),
                    "oos_LL": format_log_loss(f.get("oos_log_loss")),
                    "oos_Brier": format_brier(f.get("oos_brier")),
                    "cal_method": str(cal_d.get("method") or "—"),
                    "train_dates": str(f.get("train_date_window")),
                    "oos_dates": str(f.get("oos_date_window")),
                }
            )
        if fold_rows:
            st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)
            fv: List[float] = []
            for f in folds:
                if not isinstance(f, dict):
                    continue
                v = (f.get("calibration") or {}).get("val_log_loss_calibrated")
                if v is not None:
                    fv.append(float(v))
            if fv:
                st.caption("Val log loss (calibrated) by fold")
                st.line_chart(
                    pd.DataFrame({"fold": range(1, len(fv) + 1), "val_log_loss_cal": fv}).set_index("fold")
                )

    res = data.get("research") or {}
    if isinstance(res, dict) and res:
        st.subheader("Segmented research")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**By tour**")
            bt = res.get("by_tour") or {}
            if isinstance(bt, dict) and bt:
                st.dataframe(_dict_of_dicts_to_df(bt), use_container_width=True)
            else:
                st.caption("—")
        with c2:
            st.markdown("**By surface**")
            bs = res.get("by_surface") or {}
            if isinstance(bs, dict) and bs:
                st.dataframe(_dict_of_dicts_to_df(bs), use_container_width=True)
            else:
                st.caption("—")
        be = res.get("by_edge_bucket") or {}
        if isinstance(be, dict) and be:
            st.markdown("**By edge bucket**")
            st.dataframe(_dict_of_dicts_to_df(be), use_container_width=True)
        bo = res.get("by_odds_bucket") or {}
        if isinstance(bo, dict) and bo:
            st.markdown("**By odds bucket**")
            st.dataframe(_dict_of_dicts_to_df(bo), use_container_width=True)

    st.subheader("Optional: OOS CSV (equity / diagnostics)")
    csv_path = (oos_csv_path or "").strip()
    if csv_path and Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path)
            st.success(f"Loaded **{len(df)}** rows from `{csv_path}`")
            col_by_lower = {str(c).strip().lower(): c for c in df.columns}
            lset = set(col_by_lower.keys())
            gaps = describe_oos_csv_gaps(lset)
            for line in gaps:
                st.warning(line)
            hints = oos_csv_column_hints(lset)
            if not hints:
                st.caption("CSV has columns for equity curve and/or model–market scatter.")
            if "cumulative_pnl" in lset or "equity" in lset:
                ek = "cumulative_pnl" if "cumulative_pnl" in lset else "equity"
                col = col_by_lower[ek]
                st.caption("Bankroll / equity curve (from CSV)")
                st.line_chart(df[[col]])
            if "model_prob_a" in lset and ("market_prob_a" in lset or "implied_a" in lset):
                mp = df[col_by_lower["model_prob_a"]]
                mk = df[col_by_lower["market_prob_a"]] if "market_prob_a" in lset else df[col_by_lower["implied_a"]]
                st.caption("Model vs market probability (OOS rows)")
                st.scatter_chart(pd.DataFrame({"model_p": mp, "market_p": mk}))
        except Exception as e:
            st.warning(f"Could not read OOS CSV: {e}")
    else:
        st.caption("Pass an OOS CSV path (export from walk-forward `--out-oos-csv`) for equity curve / scatter.")

    with st.expander("Raw JSON"):
        st.json(data)


def _dict_of_dicts_to_df(d: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in d.items():
        if isinstance(v, dict):
            row = dict(v)
            row["_key"] = k
            rows.append(row)
        else:
            rows.append({"_key": k, "value": v})
    return pd.DataFrame(rows)
