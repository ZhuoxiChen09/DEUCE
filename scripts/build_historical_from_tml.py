"""
Build training CSV from TennisMyLife (TML) match CSVs.
Output schema is v3-ready: includes seed_a/b, entry_a/b, tourney_level, tourney_id,
ioc_a/b, match_num, sets_won_a/b, straight_sets_flag, tiebreak_count,
retirement_walkover_flag, long_match_flag, comeback_flag for model_features.build_match_features.

TML CSVs (from https://stats.tennismylife.org/tennis-match-database) already provide
winner_name, loser_name, winner_rank, loser_rank, winner_rank_points, loser_rank_points,
tourney_name, tourney_date, surface, score, round. No need to scrape ESPN or ATP
activity for historical match results.

Optionally merge with player_db to add serve_rating, return_rating, pressure_rating,
winloss_index, tdi_perf when available (TML does not include these).

Usage:
  python -m scripts.build_historical_from_tml -i data/2024.csv -o data/historical_2024.csv
  python -m scripts.build_historical_from_tml -i data/2020.csv data/2021.csv data/2022.csv -o data/historical_2020_2022.csv
  python -m scripts.build_historical_from_tml -i data/2024.csv -o data/historical_2024.csv --db data/player_stats.json
"""
import argparse
import csv
import logging
import random
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# TML columns required for v3 features (optional in CSV; missing -> empty in output)
V3_SOURCE_COLUMNS = [
    "winner_seed", "loser_seed", "winner_entry", "loser_entry", "tourney_level", "tourney_id",
    "winner_ioc", "loser_ioc", "match_num", "winner_hand", "loser_hand", "winner_ht", "loser_ht",
    "winner_age", "loser_age", "w_ace", "l_ace", "w_df", "l_df", "w_svpt", "l_svpt",
    "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon", "w_2ndWon", "l_2ndWon",
    "w_SvGms", "l_SvGms", "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
    "minutes", "best_of", "draw_size", "score",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from player_db import DEFAULT_DB_PATH, get_player_stats, load_db
from model_features import parse_score

OUTPUT_COLUMNS = [
    "player_a",
    "player_b",
    "winner",
    "date",
    "tournament",
    "score",
    "round",
    "surface",
    "indoor",
    "draw_size",
    "best_of",
    "minutes",
    "rank_a",
    "rank_b",
    "points_a",
    "points_b",
    "serve_rating_a",
    "serve_rating_b",
    "return_rating_a",
    "return_rating_b",
    "pressure_rating_a",
    "pressure_rating_b",
    "winloss_index_a",
    "winloss_index_b",
    "tdi_perf_a",
    "tdi_perf_b",
    "hand_a",
    "hand_b",
    "ht_a",
    "ht_b",
    "age_a",
    "age_b",
    "ace_a",
    "ace_b",
    "df_a",
    "df_b",
    "svpt_a",
    "svpt_b",
    "1stIn_a",
    "1stIn_b",
    "1stWon_a",
    "1stWon_b",
    "2ndWon_a",
    "2ndWon_b",
    "SvGms_a",
    "SvGms_b",
    "bpSaved_a",
    "bpSaved_b",
    "bpFaced_a",
    "bpFaced_b",
    "deciding_set",
    "seed_a",
    "seed_b",
    "entry_a",
    "entry_b",
    "tourney_level",
    "tourney_id",
    "ioc_a",
    "ioc_b",
    "match_num",
    "sets_won_a",
    "sets_won_b",
    "straight_sets_flag",
    "tiebreak_count",
    "retirement_walkover_flag",
    "long_match_flag",
    "comeback_flag",
]


def _safe_str(val: Any) -> str:
    if val is None or val == "":
        return ""
    s = str(val).strip()
    return s if s and s.lower() != "nan" else ""


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        s = str(val).strip().replace(",", "")
        if not s or s.lower() == "nan":
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _deciding_set_from_score(score: str, best_of: int) -> int:
    """Return 1 if match went to a deciding set (e.g. 3rd set in Bo3), else 0."""
    if not score or best_of < 3:
        return 0
    parts = score.split()
    return 1 if len(parts) >= 3 else 0


def _score_parsed_fields(score_str: str, minutes: Optional[float], swap: bool) -> Dict[str, Any]:
    """Parse score and return sets_won_a, sets_won_b, straight_sets_flag, tiebreak_count, retirement_walkover, long_match, comeback."""
    out = {
        "sets_won_a": "",
        "sets_won_b": "",
        "straight_sets_flag": "0",
        "tiebreak_count": "0",
        "retirement_walkover_flag": "0",
        "long_match_flag": "0",
        "comeback_flag": "0",
    }
    parsed = parse_score(score_str)
    sw = parsed.get("sets_winner")
    sl = parsed.get("sets_loser")
    if sw is not None and sl is not None:
        out["sets_won_a"] = str(int(sw)) if not swap else str(int(sl))
        out["sets_won_b"] = str(int(sl)) if not swap else str(int(sw))
    out["straight_sets_flag"] = "1" if parsed.get("straight_sets") else "0"
    out["tiebreak_count"] = str(int(parsed.get("tiebreak_count") or 0))
    out["retirement_walkover_flag"] = "1" if parsed.get("retirement") or parsed.get("walkover") else "0"
    out["long_match_flag"] = "1" if (minutes is not None and minutes >= 180) else "0"
    out["comeback_flag"] = "1" if parsed.get("comeback") else "0"
    return out


def tml_row_to_our_row(
    row: Dict[str, Any],
    db: Optional[List[Dict[str, Any]]],
    swap: bool,
) -> Optional[Dict[str, Any]]:
    """
    Map one TML CSV row to our training schema.
    If swap is False: player_a=winner, player_b=loser, winner=A.
    If swap is True: player_a=loser, player_b=winner, winner=B (so both classes appear in training).
    """
    winner_name = _safe_str(row.get("winner_name"))
    loser_name = _safe_str(row.get("loser_name"))
    if not winner_name or not loser_name:
        return None

    if swap:
        name_a, name_b = loser_name, winner_name
        rank_a = _safe_float(row.get("loser_rank"))
        rank_b = _safe_float(row.get("winner_rank"))
        points_a = _safe_float(row.get("loser_rank_points"))
        points_b = _safe_float(row.get("winner_rank_points"))
        winner_letter = "B"
        rec_a = get_player_stats(db, loser_name) if db else None
        rec_b = get_player_stats(db, winner_name) if db else None
        hand_a = _safe_str(row.get("loser_hand"))
        hand_b = _safe_str(row.get("winner_hand"))
        ht_a = _safe_float(row.get("loser_ht"))
        ht_b = _safe_float(row.get("winner_ht"))
        age_a = _safe_float(row.get("loser_age"))
        age_b = _safe_float(row.get("winner_age"))
        ace_a, ace_b = _safe_float(row.get("l_ace")), _safe_float(row.get("w_ace"))
        df_a, df_b = _safe_float(row.get("l_df")), _safe_float(row.get("w_df"))
        svpt_a, svpt_b = _safe_float(row.get("l_svpt")), _safe_float(row.get("w_svpt"))
        first_in_a, first_in_b = _safe_float(row.get("l_1stIn")), _safe_float(row.get("w_1stIn"))
        first_won_a, first_won_b = _safe_float(row.get("l_1stWon")), _safe_float(row.get("w_1stWon"))
        second_won_a, second_won_b = _safe_float(row.get("l_2ndWon")), _safe_float(row.get("w_2ndWon"))
        sv_gms_a, sv_gms_b = _safe_float(row.get("l_SvGms")), _safe_float(row.get("w_SvGms"))
        bp_saved_a, bp_saved_b = _safe_float(row.get("l_bpSaved")), _safe_float(row.get("w_bpSaved"))
        bp_faced_a, bp_faced_b = _safe_float(row.get("l_bpFaced")), _safe_float(row.get("w_bpFaced"))
        seed_a = _safe_float(row.get("loser_seed"))
        seed_b = _safe_float(row.get("winner_seed"))
        entry_a = _safe_str(row.get("loser_entry"))
        entry_b = _safe_str(row.get("winner_entry"))
        ioc_a = _safe_str(row.get("loser_ioc"))
        ioc_b = _safe_str(row.get("winner_ioc"))
    else:
        name_a, name_b = winner_name, loser_name
        rank_a = _safe_float(row.get("winner_rank"))
        rank_b = _safe_float(row.get("loser_rank"))
        points_a = _safe_float(row.get("winner_rank_points"))
        points_b = _safe_float(row.get("loser_rank_points"))
        winner_letter = "A"
        rec_a = get_player_stats(db, winner_name) if db else None
        rec_b = get_player_stats(db, loser_name) if db else None
        hand_a = _safe_str(row.get("winner_hand"))
        hand_b = _safe_str(row.get("loser_hand"))
        ht_a = _safe_float(row.get("winner_ht"))
        ht_b = _safe_float(row.get("loser_ht"))
        age_a = _safe_float(row.get("winner_age"))
        age_b = _safe_float(row.get("loser_age"))
        ace_a, ace_b = _safe_float(row.get("w_ace")), _safe_float(row.get("l_ace"))
        df_a, df_b = _safe_float(row.get("w_df")), _safe_float(row.get("l_df"))
        svpt_a, svpt_b = _safe_float(row.get("w_svpt")), _safe_float(row.get("l_svpt"))
        first_in_a, first_in_b = _safe_float(row.get("w_1stIn")), _safe_float(row.get("l_1stIn"))
        first_won_a, first_won_b = _safe_float(row.get("w_1stWon")), _safe_float(row.get("l_1stWon"))
        second_won_a, second_won_b = _safe_float(row.get("w_2ndWon")), _safe_float(row.get("l_2ndWon"))
        sv_gms_a, sv_gms_b = _safe_float(row.get("w_SvGms")), _safe_float(row.get("l_SvGms"))
        bp_saved_a, bp_saved_b = _safe_float(row.get("w_bpSaved")), _safe_float(row.get("l_bpSaved"))
        bp_faced_a, bp_faced_b = _safe_float(row.get("w_bpFaced")), _safe_float(row.get("l_bpFaced"))
        seed_a = _safe_float(row.get("winner_seed"))
        seed_b = _safe_float(row.get("loser_seed"))
        entry_a = _safe_str(row.get("winner_entry"))
        entry_b = _safe_str(row.get("loser_entry"))
        ioc_a = _safe_str(row.get("winner_ioc"))
        ioc_b = _safe_str(row.get("loser_ioc"))

    best_of = _safe_float(row.get("best_of")) or 3
    draw_size = _safe_float(row.get("draw_size"))
    minutes = _safe_float(row.get("minutes"))
    surface = _safe_str(row.get("surface"))
    indoor = "1" if (str(row.get("indoor") or "").strip().upper() == "I") else "0"
    deciding_set = _deciding_set_from_score(_safe_str(row.get("score")), int(best_of))
    score_str = _safe_str(row.get("score"))
    score_parsed = _score_parsed_fields(score_str, minutes, swap)
    tourney_level = _safe_str(row.get("tourney_level"))
    tourney_id = _safe_str(row.get("tourney_id"))
    match_num = _safe_float(row.get("match_num"))

    def stat(rec: Optional[Dict], key: str) -> str:
        if rec is None:
            return ""
        v = rec.get(key)
        if v is None or v == "":
            return ""
        return str(v)

    def _str(v: Any) -> str:
        if v is None or v == "":
            return ""
        return str(int(v)) if isinstance(v, float) and v == int(v) else str(v)

    return {
        "player_a": name_a,
        "player_b": name_b,
        "winner": winner_letter,
        "date": _safe_str(row.get("tourney_date")),
        "tournament": _safe_str(row.get("tourney_name")),
        "score": _safe_str(row.get("score")),
        "round": _safe_str(row.get("round")),
        "surface": surface,
        "indoor": indoor,
        "draw_size": _str(draw_size),
        "best_of": _str(best_of),
        "minutes": _str(minutes),
        "rank_a": str(int(rank_a)) if rank_a is not None else "",
        "rank_b": str(int(rank_b)) if rank_b is not None else "",
        "points_a": str(int(points_a)) if points_a is not None else "",
        "points_b": str(int(points_b)) if points_b is not None else "",
        "serve_rating_a": stat(rec_a, "serve_rating"),
        "serve_rating_b": stat(rec_b, "serve_rating"),
        "return_rating_a": stat(rec_a, "return_rating"),
        "return_rating_b": stat(rec_b, "return_rating"),
        "pressure_rating_a": stat(rec_a, "pressure_rating"),
        "pressure_rating_b": stat(rec_b, "pressure_rating"),
        "winloss_index_a": stat(rec_a, "winloss_index"),
        "winloss_index_b": stat(rec_b, "winloss_index"),
        "tdi_perf_a": stat(rec_a, "tdi_perf"),
        "tdi_perf_b": stat(rec_b, "tdi_perf"),
        "hand_a": hand_a,
        "hand_b": hand_b,
        "ht_a": _str(ht_a),
        "ht_b": _str(ht_b),
        "age_a": _str(age_a),
        "age_b": _str(age_b),
        "ace_a": _str(ace_a),
        "ace_b": _str(ace_b),
        "df_a": _str(df_a),
        "df_b": _str(df_b),
        "svpt_a": _str(svpt_a),
        "svpt_b": _str(svpt_b),
        "1stIn_a": _str(first_in_a),
        "1stIn_b": _str(first_in_b),
        "1stWon_a": _str(first_won_a),
        "1stWon_b": _str(first_won_b),
        "2ndWon_a": _str(second_won_a),
        "2ndWon_b": _str(second_won_b),
        "SvGms_a": _str(sv_gms_a),
        "SvGms_b": _str(sv_gms_b),
        "bpSaved_a": _str(bp_saved_a),
        "bpSaved_b": _str(bp_saved_b),
        "bpFaced_a": _str(bp_faced_a),
        "bpFaced_b": _str(bp_faced_b),
        "deciding_set": "1" if deciding_set else "0",
        "seed_a": _str(seed_a),
        "seed_b": _str(seed_b),
        "entry_a": entry_a,
        "entry_b": entry_b,
        "tourney_level": tourney_level,
        "tourney_id": tourney_id,
        "ioc_a": ioc_a,
        "ioc_b": ioc_b,
        "match_num": _str(match_num),
        "sets_won_a": score_parsed["sets_won_a"],
        "sets_won_b": score_parsed["sets_won_b"],
        "straight_sets_flag": score_parsed["straight_sets_flag"],
        "tiebreak_count": score_parsed["tiebreak_count"],
        "retirement_walkover_flag": score_parsed["retirement_walkover_flag"],
        "long_match_flag": score_parsed["long_match_flag"],
        "comeback_flag": score_parsed["comeback_flag"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build historical_matches CSV from TennisMyLife (TML) CSV(s)."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="TML CSV file(s), e.g. data/2020.csv data/2021.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(_REPO_ROOT / "data" / "historical_matches_tml.csv"),
        help="Output CSV path (default: data/historical_matches_tml.csv).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional player DB path to enrich serve/return/pressure/winloss/tdi. Default: no enrichment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for swapping winner/loser (so both A and B wins appear). Default: 42.",
    )
    parser.add_argument(
        "--validate-alignment",
        action="store_true",
        help="Run A/B alignment sanity checks and fail on likely column swaps.",
    )
    parser.add_argument(
        "--min-win-signal",
        type=float,
        default=0.55,
        help="Minimum fraction of (winner=A) rows where A looks stronger by rank/points (when available). Default: 0.55.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    db = None
    if args.db:
        db_path = Path(args.db)
        if not db_path.is_absolute():
            db_path = _REPO_ROOT / db_path
        db = load_db(str(db_path))
        if db:
            print(f"Loaded {len(db)} players from {db_path} for enrichment.", file=sys.stderr)
        else:
            print("Warning: empty or missing DB; rank/points from TML only.", file=sys.stderr)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict[str, Any]] = []
    for inp in args.input:
        input_path = Path(inp)
        if not input_path.is_absolute():
            input_path = _REPO_ROOT / input_path
        if not input_path.exists():
            print(f"Warning: skip missing file {input_path}.", file=sys.stderr)
            continue
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not fieldnames or "winner_name" not in fieldnames:
                print(f"Warning: {input_path} does not look like a TML CSV (no winner_name).", file=sys.stderr)
                continue
            missing_v3 = [c for c in V3_SOURCE_COLUMNS if c not in fieldnames]
            if missing_v3:
                logger.debug("TML %s: missing v3 source columns (will be empty): %s", input_path.name, missing_v3[:10])
            for row in reader:
                swap = random.random() < 0.5  # so training has both A and B wins
                out = tml_row_to_our_row(row, db, swap=swap)
                if out:
                    rows_out.append(out)
        print(f"Read {input_path.name}: total rows so far {len(rows_out)}.", file=sys.stderr)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows_out)

    if args.validate_alignment:
        # Sanity checks: if rank/points columns are swapped between A/B, these signals
        # will often invert (winner=A will look weaker most of the time).
        def sf(x: Any) -> Optional[float]:
            try:
                return float(x) if x not in (None, "") else None
            except Exception:
                return None

        usable_rank = usable_pts = 0
        good_rank = good_pts = 0
        usable_both = 0
        good_both = 0
        for r in rows_out:
            if (r.get("winner") or "").strip().upper() != "A":
                continue
            ra = sf(r.get("rank_a"))
            rb = sf(r.get("rank_b"))
            pa = sf(r.get("points_a"))
            pb = sf(r.get("points_b"))
            rank_ok = None
            pts_ok = None
            if ra is not None and rb is not None:
                usable_rank += 1
                rank_ok = (rb - ra) > 0  # rank_diff > 0 => A better
                if rank_ok:
                    good_rank += 1
            if pa is not None and pb is not None:
                usable_pts += 1
                pts_ok = (math.log1p(max(0.0, pa)) - math.log1p(max(0.0, pb))) > 0
                if pts_ok:
                    good_pts += 1
            if rank_ok is not None and pts_ok is not None:
                usable_both += 1
                if rank_ok and pts_ok:
                    good_both += 1

        def frac(g: int, u: int) -> float:
            return (g / u) if u else 0.0

        frac_rank = frac(good_rank, usable_rank)
        frac_pts = frac(good_pts, usable_pts)
        frac_both = frac(good_both, usable_both)
        min_signal = float(args.min_win_signal)
        # Only enforce when we have enough usable rows to be meaningful.
        if usable_rank >= 50 and frac_rank < min_signal:
            raise SystemExit(
                f"Alignment validation failed: among winner=A rows with rank, "
                f"only {frac_rank:.3f} have rank_diff>0 (< {min_signal}). "
                "This suggests rank_a/rank_b may be swapped."
            )
        if usable_pts >= 50 and frac_pts < min_signal:
            raise SystemExit(
                f"Alignment validation failed: among winner=A rows with points, "
                f"only {frac_pts:.3f} have log_points_diff>0 (< {min_signal}). "
                "This suggests points_a/points_b may be swapped."
            )
        # This is stricter; use only as a warning signal.
        if usable_both >= 50 and frac_both < min_signal:
            print(
                f"Warning: winner=A rows where BOTH rank+points favor A: {frac_both:.3f} (< {min_signal}). "
                "May indicate noisy ranking snapshots or alignment issues.",
                file=sys.stderr,
            )

    print(f"Wrote {len(rows_out)} rows to {output_path} (v3 schema: {len(OUTPUT_COLUMNS)} columns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
