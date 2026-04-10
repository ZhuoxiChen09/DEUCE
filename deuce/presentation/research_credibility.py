"""One-line credibility snippets for Live tab (no Streamlit dependency)."""

from __future__ import annotations

from typing import Any, Dict

from deuce.presentation.research_format import format_hit_rate, format_log_loss, format_roi_or_yield


def calibration_method_display(data: Dict[str, Any]) -> str:
    folds = data.get("folds") or []
    if folds and isinstance(folds[0], dict):
        cal = folds[0].get("calibration") or {}
        m = cal.get("method")
        if m:
            return str(m)
    return "—"


def format_credibility_caption(data: Dict[str, Any]) -> str:
    """Short line for Live markets when walk-forward JSON is loaded."""
    ll = format_log_loss(data.get("log_loss"))
    roi_s, _ = format_roi_or_yield(data.get("roi"))
    hr_s, _ = format_hit_rate(data.get("hit_rate"))
    n_oos = data.get("n_oos_predictions")
    n_str = str(n_oos) if n_oos is not None else "—"
    cal = calibration_method_display(data)
    return (
        f"**Walk-forward (loaded):** OOS log loss **{ll}** · ROI **{roi_s}** · hit rate **{hr_s}** · "
        f"n_OOS **{n_str}** · fold cal **{cal}**"
    )
