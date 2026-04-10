"""
Player-order bias audit for the tennis prediction pipeline (Core model).

Run from project root:
  python scripts/player_order_bias_audit.py

Outputs a JSON report to stdout.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _swap_match_context(mc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not mc:
        return mc
    out = dict(mc)
    # swap *_a and *_b keys
    a_keys = [k for k in out.keys() if k.endswith("_a")]
    for k_a in a_keys:
        k_b = k_a[:-2] + "_b"
        if k_b in out:
            out[k_a], out[k_b] = out[k_b], out[k_a]
    return out


def _player_stats_from_row(prefix: str, row: Dict[str, str]):
    from model_features import PlayerStats, safe_float

    def g(key: str) -> Optional[float]:
        return safe_float(row.get(f"{key}_{prefix}"))

    return PlayerStats(
        name=(row.get(f"player_{prefix}") or "").strip() or prefix.upper(),
        rank=g("rank"),
        points=g("points"),
        serve_rating=g("serve_rating"),
        return_rating=g("return_rating"),
        pressure_rating=g("pressure_rating"),
        winloss_index=g("winloss_index"),
        tdi_perf=g("tdi_perf"),
    )


def _match_context_from_hist_row(row: Dict[str, str]) -> Dict[str, Any]:
    from model_train import _match_context_from_historical_row

    # Reuse existing parsing logic (expects Dict[str, Any])
    return _match_context_from_historical_row(row)  # type: ignore[arg-type]


def _tournament_context_from_hist_row(row: Dict[str, str]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    # Minimal context used by _surface_context and tourney-level flags.
    if row.get("surface"):
        ctx["surface"] = (row.get("surface") or "").strip()
    if row.get("indoor") is not None and str(row.get("indoor")).strip() in ("1", "I", "true", "True"):
        ctx["court_type"] = "indoor"
    if row.get("best_of"):
        try:
            ctx["best_of"] = float(row.get("best_of") or 3)
        except Exception:
            pass
    if row.get("draw_size"):
        try:
            ctx["draw_size"] = float(row.get("draw_size") or 64)
        except Exception:
            pass
    if row.get("round"):
        ctx["round"] = (row.get("round") or "").strip()
    # Tier used by _tourney_level_flags in model_features via mc['tourney_level'] too
    return ctx


def _predict_with_features(
    *,
    bundle: Dict[str, Any],
    feature_names: List[str],
    player_a,
    player_b,
    tournament_name: Optional[str],
    catalog: Any,
    match_context: Optional[Dict[str, Any]],
    feature_set: str = "core",
) -> Tuple[float, Dict[str, float], List[float]]:
    import numpy as np
    from model_features import build_match_features, feature_dict_from_match_features
    from model_predict import predict_match_probability

    res = predict_match_probability(
        player_a,
        player_b,
        model_core=bundle,
        model=None,
        model_enhanced=None,
        tournament_name=tournament_name,
        catalog=catalog,
        match_context=match_context,
    )
    # Rebuild feature dict/vector exactly as used (core)
    mf = build_match_features(
        player_a,
        player_b,
        tournament_context=None,
        medians=bundle.get("medians") or {},
        match_context=match_context,
        feature_set=feature_set,
    )
    fv = feature_dict_from_match_features(mf)
    vec = [fv.get(n, 0.0) for n in feature_names]
    # (Vector here is raw; model scales internally. Keep raw for audit output.)
    return float(res.model_prob_a), fv, vec


def main() -> int:
    from model_predict import load_model
    from model_features import build_player_stats_from_datasets
    from player_db import DEFAULT_DB_PATH, load_db

    model_path = ROOT / "artifacts" / "tennis_model_core.pkl"
    if not model_path.exists():
        print(json.dumps({"error": f"Missing model: {model_path}"}))
        return 1

    bundle, feature_names, _ = load_model(str(model_path))
    if not isinstance(bundle, dict) or bundle.get("version") not in (2, 3):
        print(json.dumps({"error": "Core model bundle not found or not a v2/v3 bundle"}))
        return 1

    # ---- (1) Specific symmetry test using player DB (app path) ----
    db = load_db(DEFAULT_DB_PATH)
    name1 = "Alejandro Davidovich Fokina"
    name2 = "Andrea Pellegrino"
    stats_a = build_player_stats_from_datasets(name1, player_db=db)
    stats_b = build_player_stats_from_datasets(name2, player_db=db)
    # No tournament context provided by user; keep None (exact same context both cases)
    tournament_name = None
    catalog = None
    mc: Dict[str, Any] = {}

    p1, fv1, rawvec1 = _predict_with_features(
        bundle=bundle,
        feature_names=feature_names,
        player_a=stats_a,
        player_b=stats_b,
        tournament_name=tournament_name,
        catalog=catalog,
        match_context=mc or None,
        feature_set="core",
    )
    p2, fv2, rawvec2 = _predict_with_features(
        bundle=bundle,
        feature_names=feature_names,
        player_a=stats_b,
        player_b=stats_a,
        tournament_name=tournament_name,
        catalog=catalog,
        match_context=_swap_match_context(mc) or None,
        feature_set="core",
    )
    symmetry_err_specific = abs(p1 - (1.0 - p2))

    # ---- (5) Slot-bias detection: 20 random matchups from historical csv ----
    hist_path = ROOT / "data" / "historical_2026_v3.csv"
    swap_results: List[Dict[str, Any]] = []
    if hist_path.exists():
        with open(hist_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        rng = random.Random(42)
        sample = rows if len(rows) <= 20 else rng.sample(rows, 20)
        for r in sample:
            pa = _player_stats_from_row("a", r)
            pb = _player_stats_from_row("b", r)
            mc_r = _match_context_from_hist_row(r)
            tournament_name_r = (r.get("tournament") or "").strip() or None
            p_ab, _, _ = _predict_with_features(
                bundle=bundle,
                feature_names=feature_names,
                player_a=pa,
                player_b=pb,
                tournament_name=tournament_name_r,
                catalog=None,
                match_context=mc_r if any(mc_r.values()) else None,
                feature_set="core",
            )
            p_ba, _, _ = _predict_with_features(
                bundle=bundle,
                feature_names=feature_names,
                player_a=pb,
                player_b=pa,
                tournament_name=tournament_name_r,
                catalog=None,
                match_context=_swap_match_context(mc_r) if any(mc_r.values()) else None,
                feature_set="core",
            )
            err = abs(p_ab - (1.0 - p_ba))
            swap_results.append(
                {
                    "player_a": pa.name,
                    "player_b": pb.name,
                    "tournament": (r.get("tournament") or "").strip(),
                    "p_a_wins": p_ab,
                    "p_after_swap_a_wins": p_ba,
                    "symmetry_error": err,
                }
            )

    swap_results_sorted = sorted(swap_results, key=lambda x: -x["symmetry_error"])
    avg_abs_sym_err = (
        sum(x["symmetry_error"] for x in swap_results) / len(swap_results)
        if swap_results
        else None
    )

    out = {
        "model_path": str(model_path),
        "feature_count": len(feature_names),
        "specific_symmetry_test": {
            "case_1": {
                "player_a": name1,
                "player_b": name2,
                "p_a_wins": p1,
                "p_b_wins": 1.0 - p1,
                "raw_feature_values": fv1,
                "raw_feature_vector_in_model_order": rawvec1,
            },
            "case_2": {
                "player_a": name2,
                "player_b": name1,
                "p_a_wins": p2,
                "p_b_wins": 1.0 - p2,
                "raw_feature_values": fv2,
                "raw_feature_vector_in_model_order": rawvec2,
            },
            "symmetry_error_abs": symmetry_err_specific,
            "approximately_flips": symmetry_err_specific < 0.05,
            "favored_player_case_1": name1 if p1 >= 0.5 else name2,
            "favored_player_case_2": name2 if p2 >= 0.5 else name1,
        },
        "random_swap_test": {
            "n": len(swap_results),
            "avg_abs_symmetry_error": avg_abs_sym_err,
            "worst_10": swap_results_sorted[:10],
        },
        "notes": {
            "context_used_for_specific_test": "empty tournament_context and empty match_context (DB-based ranks/points only, no seeds/entry/tourney_id).",
            "symmetry_expected": "For a fully symmetric model, swapping players should yield p_swap ≈ 1 - p_original. Deviations indicate slot bias or non-swapped features.",
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

