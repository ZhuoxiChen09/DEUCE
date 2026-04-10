"""
Aggregate research metrics from precomputed rows (model prob, outcome, optional market).

Each row: model_prob_a, outcome_a_won (0/1), optional market_price_a/b, tour, surface, ...
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deuce.backtest.engine import BacktestConfig, run_backtest


@dataclass
class ResearchReport:
    n: int
    log_loss: float
    brier: float
    roi: float
    yield_per_bet: float
    max_drawdown: float
    avg_edge: float
    hit_rate: float
    by_tour: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_surface: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_edge_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_odds_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _log_loss_brier(probs: List[float], y: List[int]) -> tuple[float, float]:
    ll = 0.0
    br = 0.0
    for p, yi in zip(probs, y):
        p = max(1e-15, min(1.0 - 1e-15, p))
        p_win = p if yi == 1 else (1.0 - p)
        ll -= math.log(p_win)
        br += (p - yi) ** 2
    n = len(probs)
    return (ll / n if n else 0.0), (br / n if n else 0.0)


def _agg_ll_brier(groups: Dict[str, List[tuple[float, int]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, pairs in groups.items():
        pp = [x[0] for x in pairs]
        yy = [x[1] for x in pairs]
        l, b = _log_loss_brier(pp, yy)
        out[k] = {"log_loss": l, "brier": b, "n": float(len(pairs))}
    return out


def research_report_from_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    backtest_cfg: Optional[BacktestConfig] = None,
) -> ResearchReport:
    """Probability metrics (log loss, Brier) and segmented backtest PnL metrics."""
    probs: List[float] = []
    ys: List[int] = []
    bt_rows: List[Dict[str, Any]] = []
    by_tour: Dict[str, List[tuple[float, int]]] = {}
    by_surface: Dict[str, List[tuple[float, int]]] = {}

    for r in rows:
        try:
            p = float(r.get("model_prob_a", r.get("prob_a", 0.5)))
            y = int(r.get("outcome_a_won", r.get("y", 0)))
        except (TypeError, ValueError):
            continue
        probs.append(p)
        ys.append(y)
        br = dict(r)
        br["model_prob_a"] = p
        br["outcome_a_won"] = y
        bt_rows.append(br)
        t = str(r.get("tour", "UNK"))
        s = str(r.get("surface", "UNK"))
        by_tour.setdefault(t, []).append((p, y))
        by_surface.setdefault(s, []).append((p, y))

    if not probs:
        return ResearchReport(
            n=0, log_loss=0.0, brier=0.0, roi=0.0, yield_per_bet=0.0,
            max_drawdown=0.0, avg_edge=0.0, hit_rate=0.0,
        )

    ll, br = _log_loss_brier(probs, ys)
    cfg = backtest_cfg or BacktestConfig(edge_threshold=0.0)
    res_bt = run_backtest(bt_rows, config=cfg)

    return ResearchReport(
        n=len(probs),
        log_loss=ll,
        brier=br,
        roi=res_bt.roi,
        yield_per_bet=res_bt.yield_per_bet,
        max_drawdown=res_bt.max_drawdown,
        avg_edge=res_bt.avg_edge,
        hit_rate=res_bt.hit_rate,
        by_tour=_agg_ll_brier(by_tour),
        by_surface=_agg_ll_brier(by_surface),
        by_edge_bucket=res_bt.segments_edge_bucket,
        by_odds_bucket=res_bt.segments_odds_bucket,
    )


def research_report_from_csv(path: str, *, backtest_cfg: Optional[BacktestConfig] = None) -> ResearchReport:
    """Load CSV and build a ``ResearchReport``."""
    p = Path(path)
    with open(p, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if v is not None and k in (
                "model_prob_a", "prob_a", "outcome_a_won", "y",
                "market_price_a", "market_price_b",
            ):
                try:
                    if k.startswith("outcome") or k == "y":
                        r[k] = int(float(v)) if v != "" else 0
                    else:
                        r[k] = float(v) if v != "" else None
                except (TypeError, ValueError):
                    pass
    return research_report_from_rows(rows, backtest_cfg=backtest_cfg)


def report_to_json_dict(rep: ResearchReport) -> Dict[str, Any]:
    return asdict(rep)
