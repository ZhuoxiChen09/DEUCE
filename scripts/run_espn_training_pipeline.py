"""
Pipeline: fetch ESPN match results for 2020-2025, build historical CSV with player stats, train with 80/20 split.

Usage:
  python -m scripts.run_espn_training_pipeline                    # 2020-2025, 80/20
  python -m scripts.run_espn_training_pipeline --start 2023 --end 2025
  python -m scripts.run_espn_training_pipeline --skip-fetch       # use existing match list
  python -m scripts.run_espn_training_pipeline --max-dates-per-season 40  # limit requests (e.g. for testing)
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MATCH_LIST_FIELDS = ["player_a", "player_b", "winner", "date", "tournament", "score", "round"]


def fetch_matches_for_years(
    start_year: int,
    end_year: int,
    max_dates_per_season: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch ESPN matches for each season and combine (dedupe by date, player_a, player_b)."""
    from espn_tennis import fetch_espn_matches_for_season

    all_matches: List[Dict[str, Any]] = []
    seen: set = set()
    for season in range(start_year, end_year + 1):
        print(f"Fetching ESPN matches for season {season}...", file=sys.stderr)
        matches = fetch_espn_matches_for_season(
            season,
            atp_only=True,
            max_dates=max_dates_per_season,
        )
        for m in matches:
            key = (m.get("date") or "", tuple(sorted([m["player_a"], m["player_b"]])))
            if key not in seen:
                seen.add(key)
                all_matches.append(m)
        print(f"  Season {season}: {len(matches)} matches (total so far: {len(all_matches)})", file=sys.stderr)
    return all_matches


def write_match_list(matches: List[Dict[str, Any]], out_path: Path) -> None:
    """Write match list CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "player_a": m["player_a"],
            "player_b": m["player_b"],
            "winner": m["winner"],
            "date": m.get("date") or "",
            "tournament": m.get("tournament") or "",
            "score": m.get("score") or "",
            "round": m.get("round") or "",
        }
        for m in matches
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_LIST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ESPN match data → historical CSV → train model (80/20 split)."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2020,
        help="First season year (default: 2020).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2025,
        help="Last season year inclusive (default: 2025).",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip ESPN fetch; use existing match list CSV.",
    )
    parser.add_argument(
        "--max-dates-per-season",
        type=int,
        default=None,
        help="Max scoreboard dates per season (default: all). Use e.g. 40 for faster test run.",
    )
    parser.add_argument(
        "--match-list",
        default=str(_REPO_ROOT / "data" / "match_list_2020_2025.csv"),
        help="Path to match list CSV (default: data/match_list_2020_2025.csv).",
    )
    parser.add_argument(
        "--historical",
        default=str(_REPO_ROOT / "data" / "historical_matches_2020_2025.csv"),
        help="Path to historical CSV with stats (default: data/historical_matches_2020_2025.csv).",
    )
    parser.add_argument(
        "--model-out",
        default=str(_REPO_ROOT / "artifacts" / "tennis_model.pkl"),
        help="Output model path (default: artifacts/tennis_model.pkl).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Validation fraction for 80/20 split (default: 0.2).",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Optional tournament_catalog.json for sample weights.",
    )
    args = parser.parse_args()

    match_list_path = Path(args.match_list)
    if not match_list_path.is_absolute():
        match_list_path = _REPO_ROOT / match_list_path
    historical_path = Path(args.historical)
    if not historical_path.is_absolute():
        historical_path = _REPO_ROOT / historical_path
    model_out_path = Path(args.model_out)
    if not model_out_path.is_absolute():
        model_out_path = _REPO_ROOT / model_out_path

    if not args.skip_fetch:
        matches = fetch_matches_for_years(
            args.start,
            args.end,
            max_dates_per_season=args.max_dates_per_season,
        )
        if not matches:
            print("No matches fetched. Aborting.", file=sys.stderr)
            return 1
        write_match_list(matches, match_list_path)
        print(f"Wrote {len(matches)} matches to {match_list_path}.", file=sys.stderr)
    else:
        if not match_list_path.exists():
            print(f"Match list not found: {match_list_path}. Run without --skip-fetch.", file=sys.stderr)
            return 1
        print(f"Using existing match list: {match_list_path}.", file=sys.stderr)

    # Build historical CSV from match list + player DB
    build_cmd = [
        sys.executable,
        "-m",
        "scripts.build_historical_from_db",
        "-i",
        str(match_list_path),
        "-o",
        str(historical_path),
    ]
    print("Building historical CSV with player stats...", file=sys.stderr)
    r = subprocess.run(build_cmd, cwd=str(_REPO_ROOT))
    if r.returncode != 0:
        print("build_historical_from_db failed.", file=sys.stderr)
        return r.returncode

    # Train with 80/20 split
    train_cmd = [
        sys.executable,
        "model_train.py",
        "--data",
        str(historical_path),
        "--out",
        str(model_out_path),
        "--val-frac",
        str(args.val_frac),
    ]
    if args.catalog:
        train_cmd.extend(["--catalog", args.catalog])
    print(f"Training model (val_frac={args.val_frac})...", file=sys.stderr)
    r = subprocess.run(train_cmd, cwd=str(_REPO_ROOT))
    if r.returncode != 0:
        print("model_train failed.", file=sys.stderr)
        return r.returncode

    print("Pipeline done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
