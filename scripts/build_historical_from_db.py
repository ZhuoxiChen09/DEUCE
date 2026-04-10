"""
Build a full historical_matches CSV from a simple match list (player_a, player_b, winner)
by looking up stats from the player database (data/player_stats.json).

Input CSV: must have player_a, player_b, winner (A or B). Optional: date, tournament, score, round (passed through).
Output CSV: same schema as data/historical_matches_sample.csv with stat columns filled from DB.

Usage:
  python -m scripts.build_historical_from_db -i data/match_list.csv -o data/historical_matches.csv
  python -m scripts.build_historical_from_db -i data/match_list.csv  # writes to data/historical_matches.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Allow running from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from player_db import DEFAULT_DB_PATH, get_player_stats, load_db

OUTPUT_COLUMNS = [
    "player_a",
    "player_b",
    "winner",
    "date",
    "tournament",
    "score",
    "round",
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
]


def _stat(rec: Optional[Dict[str, Any]], key: str) -> str:
    if rec is None:
        return ""
    v = rec.get(key)
    if v is None or v == "":
        return ""
    return v


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build historical_matches.csv from match list + player DB."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV with player_a, player_b, winner (A or B).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(_REPO_ROOT / "data" / "historical_matches.csv"),
        help="Output CSV path (default: data/historical_matches.csv).",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help="Path to player stats JSON (default: data/player_stats.json).",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = _REPO_ROOT / db_path
    db = load_db(str(db_path))
    if not db:
        print(f"Warning: no players in DB at {db_path}. Stats columns will be empty.", file=sys.stderr)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = _REPO_ROOT / input_path
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "player_a" not in reader.fieldnames or "player_b" not in reader.fieldnames or "winner" not in reader.fieldnames:
            print("Error: input CSV must have player_a, player_b, winner.", file=sys.stderr)
            return 1
        for row in reader:
            player_a = (row.get("player_a") or "").strip()
            player_b = (row.get("player_b") or "").strip()
            winner = (row.get("winner") or "").strip().upper()
            if not player_a or not player_b:
                continue
            if winner not in ("A", "B"):
                winner = ""
            rec_a = get_player_stats(db, player_a) if db else None
            rec_b = get_player_stats(db, player_b) if db else None
            out = {
                "player_a": player_a,
                "player_b": player_b,
                "winner": winner,
                "date": (row.get("date") or "").strip(),
                "tournament": (row.get("tournament") or "").strip(),
                "score": (row.get("score") or "").strip(),
                "round": (row.get("round") or "").strip(),
                "rank_a": _stat(rec_a, "rank"),
                "rank_b": _stat(rec_b, "rank"),
                "points_a": _stat(rec_a, "points"),
                "points_b": _stat(rec_b, "points"),
                "serve_rating_a": _stat(rec_a, "serve_rating"),
                "serve_rating_b": _stat(rec_b, "serve_rating"),
                "return_rating_a": _stat(rec_a, "return_rating"),
                "return_rating_b": _stat(rec_b, "return_rating"),
                "pressure_rating_a": _stat(rec_a, "pressure_rating"),
                "pressure_rating_b": _stat(rec_b, "pressure_rating"),
                "winloss_index_a": _stat(rec_a, "winloss_index"),
                "winloss_index_b": _stat(rec_b, "winloss_index"),
                "tdi_perf_a": _stat(rec_a, "tdi_perf"),
                "tdi_perf_b": _stat(rec_b, "tdi_perf"),
            }
            rows_out.append(out)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
