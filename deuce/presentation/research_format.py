"""
Robust display for walk-forward / research JSON values.

DEUCE pipeline stores ROI, yield, hit_rate, avg_edge as proportions in [0, 1] (or small multiples).
External files may use 0–100 percent scale. We never silently assume the wrong scale.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def format_log_loss(v: Any) -> str:
    x = as_float(v)
    return f"{x:.4f}" if x is not None else "—"


def format_brier(v: Any) -> str:
    x = as_float(v)
    return f"{x:.4f}" if x is not None else "—"


def format_roi_or_yield(v: Any) -> Tuple[str, str]:
    """
    Returns (display, interpretation).

    |x| <= 1.5 → treat as proportion of stake → show as %.
    1.5 < |x| <= 100 → treat as already on 0–100 percent scale.
    Else → raw with explicit note.
    """
    x = as_float(v)
    if x is None:
        return "—", ""
    ax = abs(x)
    if ax <= 1.5:
        return f"{x * 100:.2f}%", "proportion→%"
    if ax <= 100.0:
        return f"{x:.2f}%", "0–100 scale"
    return f"{x:.4f}", "raw (check units)"


def format_hit_rate(v: Any) -> Tuple[str, str]:
    x = as_float(v)
    if x is None:
        return "—", ""
    if 0.0 <= x <= 1.0 + 1e-6:
        return f"{x * 100:.1f}%", "proportion"
    if 0.0 < x <= 100.0 + 1e-6:
        return f"{x:.1f}%", "0–100 scale"
    return f"{x:.4f}", "raw (check units)"


def format_avg_edge(v: Any) -> Tuple[str, str]:
    """Mean |edge| in backtest rows; usually proportion (probability points)."""
    x = as_float(v)
    if x is None:
        return "—", ""
    if abs(x) <= 1.5:
        return f"{x * 100:.2f} pp", "proportion→pp"
    if abs(x) <= 100.0:
        return f"{x:.2f} pp", "0–100 scale"
    return f"{x:.4f}", "raw"


def format_max_drawdown(v: Any) -> Tuple[str, str]:
    """
    Pipeline uses absolute drawdown in PnL units from run_backtest (not necessarily [0,1]).

    If value is in [0, 1], treat as fraction of peak; else numeric.
    """
    x = as_float(v)
    if x is None:
        return "—", ""
    if x < 0:
        return f"{x:.4f}", "negative (unusual)"
    if 0.0 <= x <= 1.0 + 1e-6:
        return f"{x * 100:.1f}% peak", "fraction of peak"
    return f"{x:.2f}", "PnL units"


def format_calibration_scalar(v: Any) -> str:
    x = as_float(v)
    return f"{x:.6f}" if x is not None else "—"


def headline_scale_caption(notes: list[str]) -> str:
    uniq = [n for n in dict.fromkeys(notes) if n]
    if not uniq:
        return ""
    return " · ".join(f"*{t}*" for t in uniq)
