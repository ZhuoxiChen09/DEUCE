"""
Dataset alignment audit for historical training CSVs.

Focus: player_a/player_b/winner alignment with rank/points/seed fields and derived diffs.

Run from project root:
  python scripts/dataset_alignment_audit.py --csv data/historical_2026_v3.csv --sample 50

Writes JSON to stdout and a sample table CSV next to the input file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sf(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _log_points_diff(pa: Optional[float], pb: Optional[float]) -> Optional[float]:
    if pa is None or pb is None:
        return None
    return math.log1p(max(0.0, float(pa))) - math.log1p(max(0.0, float(pb)))


def _seed_diff(seed_a: Optional[float], seed_b: Optional[float]) -> Optional[float]:
    if seed_a is None or seed_b is None:
        return None
    if seed_a <= 0 or seed_b <= 0:
        return None
    return float(seed_b) - float(seed_a)


def _rank_diff(rank_a: Optional[float], rank_b: Optional[float]) -> Optional[float]:
    if rank_a is None or rank_b is None:
        return None
    return float(rank_b) - float(rank_a)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit A/B alignment in historical training CSV.")
    parser.add_argument("--csv", required=True, help="Path to historical training CSV (v3 schema).")
    parser.add_argument("--sample", type=int, default=50, help="Number of rows to sample for row-level table.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    rng = random.Random(args.seed)
    sample_n = min(args.sample, n)
    sample_rows = rng.sample(rows, sample_n) if n else []

    def label(row: Dict[str, str]) -> int:
        return 1 if (row.get("winner") or "").strip().upper() == "A" else 0

    # Row-level sample table
    sample_table: List[Dict[str, Any]] = []
    for r in sample_rows:
        ra = _sf(r.get("rank_a"))
        rb = _sf(r.get("rank_b"))
        pa = _sf(r.get("points_a"))
        pb = _sf(r.get("points_b"))
        sa = _sf(r.get("seed_a"))
        sb = _sf(r.get("seed_b"))
        y = label(r)

        rd = _rank_diff(ra, rb)
        lpd = _log_points_diff(pa, pb)
        sd = _seed_diff(sa, sb)

        rank_cons = None
        if ra is not None and rb is not None:
            a_better = ra < rb
            rank_cons = (y == 1 and a_better) or (y == 0 and (not a_better))
        pts_cons = None
        if pa is not None and pb is not None:
            a_more = pa > pb
            pts_cons = (y == 1 and a_more) or (y == 0 and (not a_more))
        seed_cons = None
        if sd is not None:
            a_better = sa < sb  # lower seed is better
            seed_cons = (y == 1 and a_better) or (y == 0 and (not a_better))

        sample_table.append(
            {
                "player_a": r.get("player_a"),
                "player_b": r.get("player_b"),
                "winner": r.get("winner"),
                "rank_a": ra,
                "rank_b": rb,
                "points_a": pa,
                "points_b": pb,
                "seed_a": sa,
                "seed_b": sb,
                "rank_diff_computed": rd,
                "log_points_diff_computed": lpd,
                "seed_diff_computed": sd,
                "rank_consistent_with_label": rank_cons,
                "points_consistent_with_label": pts_cons,
                "seed_consistent_with_label": seed_cons,
            }
        )

    # Full-dataset contradiction rates
    def _rate(cond_rows: List[Dict[str, str]], pred) -> float:
        if not cond_rows:
            return 0.0
        return 100.0 * sum(1 for r in cond_rows if pred(r)) / len(cond_rows)

    def a_better_rank(r: Dict[str, str]) -> Optional[bool]:
        ra = _sf(r.get("rank_a"))
        rb = _sf(r.get("rank_b"))
        if ra is None or rb is None:
            return None
        return ra < rb

    def a_more_points(r: Dict[str, str]) -> Optional[bool]:
        pa = _sf(r.get("points_a"))
        pb = _sf(r.get("points_b"))
        if pa is None or pb is None:
            return None
        return pa > pb

    def a_better_seed(r: Dict[str, str]) -> Optional[bool]:
        sa = _sf(r.get("seed_a"))
        sb = _sf(r.get("seed_b"))
        if sa is None or sb is None or sa <= 0 or sb <= 0:
            return None
        return sa < sb

    rows_rank = [r for r in rows if a_better_rank(r) is not None]
    rows_pts = [r for r in rows if a_more_points(r) is not None]
    rows_seed = [r for r in rows if a_better_seed(r) is not None]

    def a_loses(r: Dict[str, str]) -> bool:
        return label(r) == 0

    # Percent where A has better metric but loses
    pct_rank_better_but_loses = _rate(rows_rank, lambda r: bool(a_better_rank(r)) and a_loses(r))
    pct_pts_more_but_loses = _rate(rows_pts, lambda r: bool(a_more_points(r)) and a_loses(r))
    pct_seed_better_but_loses = _rate(rows_seed, lambda r: bool(a_better_seed(r)) and a_loses(r))

    rows_wA = [r for r in rows if label(r) == 1]
    rows_wB = [r for r in rows if label(r) == 0]

    def pct_in_subset(sub: List[Dict[str, str]], metric_fn) -> float:
        usable = [r for r in sub if metric_fn(r) is not None]
        return _rate(usable, lambda r: bool(metric_fn(r)) and a_loses(r))

    by_winner = {
        "winner_A": {
            "pct_A_better_rank_but_loses": pct_in_subset(rows_wA, a_better_rank),
            "pct_A_more_points_but_loses": pct_in_subset(rows_wA, a_more_points),
            "pct_A_better_seed_but_loses": pct_in_subset(rows_wA, a_better_seed),
        },
        "winner_B": {
            "pct_A_better_rank_but_loses": pct_in_subset(rows_wB, a_better_rank),
            "pct_A_more_points_but_loses": pct_in_subset(rows_wB, a_more_points),
            "pct_A_better_seed_but_loses": pct_in_subset(rows_wB, a_better_seed),
        },
    }

    # Univariate LR sign checks (rank_diff, log_points_diff, seed_diff)
    uni = {}
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        def uni_fit(get_x):
            Xv = []
            yv = []
            for r in rows:
                x = get_x(r)
                if x is None:
                    continue
                Xv.append([float(x)])
                yv.append(label(r))
            if len(yv) < 20:
                return None
            Xv = np.array(Xv, dtype=float)
            yv = np.array(yv, dtype=int)
            Xtr, Xte, ytr, yte = train_test_split(Xv, yv, test_size=0.2, random_state=42, stratify=yv)
            m = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            auc = float(roc_auc_score(yte, p))
            return {"coef": float(m.coef_[0][0]), "intercept": float(m.intercept_[0]), "auc": auc, "n": int(len(yv))}

        uni["rank_diff"] = uni_fit(lambda r: _rank_diff(_sf(r.get("rank_a")), _sf(r.get("rank_b"))))
        uni["log_points_diff"] = uni_fit(lambda r: _log_points_diff(_sf(r.get("points_a")), _sf(r.get("points_b"))))
        uni["seed_diff"] = uni_fit(lambda r: _seed_diff(_sf(r.get("seed_a")), _sf(r.get("seed_b"))))
    except Exception as e:
        uni["error"] = str(e)

    # Mirror consistency check: detect exact duplicate pairs where player_a/b swapped and date/tournament/score same.
    # (This is heuristic; true identity ideally includes tourney_id + match_num.)
    key_to_row = {}
    mirror_violations = 0
    mirror_checked = 0
    for r in rows:
        k = (r.get("date"), r.get("tournament"), r.get("score"), r.get("player_a"), r.get("player_b"))
        key_to_row[k] = r
    for r in rows:
        k_swap = (r.get("date"), r.get("tournament"), r.get("score"), r.get("player_b"), r.get("player_a"))
        other = key_to_row.get(k_swap)
        if not other:
            continue
        mirror_checked += 1
        # check swapped ranks/points/seeds and inverted winner label
        ok = True
        if (r.get("rank_a") or "") != (other.get("rank_b") or ""):
            ok = False
        if (r.get("rank_b") or "") != (other.get("rank_a") or ""):
            ok = False
        if (r.get("points_a") or "") != (other.get("points_b") or ""):
            ok = False
        if (r.get("points_b") or "") != (other.get("points_a") or ""):
            ok = False
        if (r.get("seed_a") or "") != (other.get("seed_b") or ""):
            ok = False
        if (r.get("seed_b") or "") != (other.get("seed_a") or ""):
            ok = False
        if label(r) == label(other):
            ok = False
        if not ok:
            mirror_violations += 1

    # Write sample table CSV
    out_csv = path.with_suffix(path.suffix + ".alignment_sample.csv")
    if sample_table:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sample_table[0].keys()))
            w.writeheader()
            w.writerows(sample_table)

    out = {
        "csv": str(path),
        "n_rows": n,
        "sample_n": sample_n,
        "sample_table_csv": str(out_csv),
        "contradictions": {
            "pct_A_better_rank_but_loses": pct_rank_better_but_loses,
            "pct_A_more_points_but_loses": pct_pts_more_but_loses,
            "pct_A_better_seed_but_loses": pct_seed_better_but_loses,
            "by_winner": by_winner,
        },
        "univariate_checks": uni,
        "mirror_consistency": {
            "pairs_checked": mirror_checked,
            "violations": mirror_violations,
            "violation_rate_pct": (100.0 * mirror_violations / mirror_checked) if mirror_checked else None,
        },
        "formulas": {
            "y": "1 if winner == 'A' else 0",
            "rank_diff": "rank_b - rank_a",
            "log_points_diff": "log(1+points_a) - log(1+points_b)",
            "seed_diff": "seed_b - seed_a (only when both seeds > 0)",
            "sign_meaning": {
                "rank_diff>0": "A has better (lower) rank number than B",
                "log_points_diff>0": "A has more points than B",
                "seed_diff>0": "A has better (lower) seed than B",
            },
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

