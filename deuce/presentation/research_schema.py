"""Expected keys for walk-forward JSON + OOS CSV (docs + validation)."""

from __future__ import annotations

from typing import Any, Dict, List, Set

# From aggregate_report_to_json (walk_forward.py)
EXPECTED_JSON_KEYS_PRIMARY: List[str] = [
    "mode",
    "n_folds",
    "n_oos_predictions",
    "log_loss",
    "brier",
    "roi",
    "yield_per_bet",
    "max_drawdown",
    "avg_edge",
    "hit_rate",
    "calibration_mean_val_ll_calibrated",
    "calibration_summary",
    "bankroll",
    "research",
    "folds",
]

OPTIONAL_JSON_GROUPS = (
    "Segments live under **research**: by_tour, by_surface, by_edge_bucket, by_odds_bucket."
)

OOS_CSV_COLUMNS_HELP = """
**Equity curve:** `cumulative_pnl` or `equity` (one series per row).

**Scatter (model vs market):** `model_prob_a` plus one of `market_prob_a`, `implied_a`.

Other useful columns if present: `outcome_a_won`, `fold_index`, `edge`, `tour`, `surface`.
""".strip()

WALK_FORWARD_JSON_FIELDS_HELP = """
**Headline OOS:** `log_loss`, `brier`, `roi`, `yield_per_bet`, `max_drawdown`, `hit_rate`, `avg_edge`,
`n_oos_predictions`, `n_folds`, `mode`.

**Bankroll block:** `bankroll` with `roi`, `yield_per_bet`, `max_drawdown`, `n_bets`.

**Calibration:** `calibration_summary` and/or `calibration_mean_val_ll_calibrated`; per fold `folds[].calibration.method`,
`val_log_loss_calibrated`, `oos_log_loss`, `oos_brier`.

**Segments:** `research.by_tour`, `by_surface`, `by_edge_bucket`, `by_odds_bucket`.
""".strip()


def missing_json_keys(data: Dict[str, Any]) -> List[str]:
    if not data:
        return list(EXPECTED_JSON_KEYS_PRIMARY)
    have = set(data.keys())
    return [k for k in EXPECTED_JSON_KEYS_PRIMARY if k not in have]


def oos_csv_column_hints(df_columns: Set[str]) -> Dict[str, List[str]]:
    """Return missing *useful* column groups for warnings."""
    c = {x.lower() for x in df_columns}
    out: Dict[str, List[str]] = {}
    if not ({"cumulative_pnl"} & c or {"equity"} & c):
        out["equity_curve"] = ["cumulative_pnl or equity"]
    if "model_prob_a" not in c:
        out["scatter_model"] = ["model_prob_a"]
    if "market_prob_a" not in c and "implied_a" not in c:
        out["scatter_market"] = ["market_prob_a or implied_a"]
    return out


def describe_oos_csv_gaps(df_columns: Set[str]) -> List[str]:
    """Human-readable lines for UI warnings when optional OOS CSV is thin."""
    hints = oos_csv_column_hints(df_columns)
    lines: List[str] = []
    if "equity_curve" in hints:
        lines.append(
            "No **cumulative_pnl** or **equity** column — equity curve chart is skipped."
        )
    if "scatter_model" in hints or "scatter_market" in hints:
        need = []
        if "scatter_model" in hints:
            need.append("`model_prob_a`")
        if "scatter_market" in hints:
            need.append("`market_prob_a` or `implied_a`")
        lines.append("Model vs market scatter skipped — add " + " and ".join(need) + ".")
    return lines
