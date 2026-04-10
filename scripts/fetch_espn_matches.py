"""
Fetch ATP match results from ESPN for a season: uses schedule to get dates with matches,
then scoreboard per date. Writes match list CSV with player_a, player_b, winner, date,
tournament, score (and round) for use with build_historical_from_db.py. No Playwright required.

Usage:
  python -m scripts.fetch_espn_matches --season 2025 --out data/match_list.csv
  python -m scripts.fetch_espn_matches --season 2025 --max-dates 30  # limit to 30 scoreboard requests
"""
import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from espn_tennis import fetch_espn_matches_for_season, get_scoreboard_matches

MATCH_LIST_FIELDS = ["player_a", "player_b", "winner", "date", "tournament", "score", "round"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch ATP matches from ESPN scoreboard by season."
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season year (e.g. 2025).",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "data" / "match_list.csv"),
        help="Output CSV path (player_a, player_b, winner, date, tournament, score, round).",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Max number of scoreboard dates to fetch (default: all from schedule).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Single date YYYYMMDD to fetch (overrides season).",
    )
    args = parser.parse_args()

    if args.date:
        matches = get_scoreboard_matches(args.date)
        print(f"Fetched {len(matches)} matches for {args.date}.", file=sys.stderr)
    else:
        matches = fetch_espn_matches_for_season(
            args.season,
            atp_only=True,
            max_dates=args.max_dates,
        )
        print(f"Fetched {len(matches)} matches for season {args.season}.", file=sys.stderr)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []
    for m in matches:
        key = tuple(sorted([m["player_a"], m["player_b"]]))
        if key not in seen:
            seen.add(key)
            rows.append({
                "player_a": m["player_a"],
                "player_b": m["player_b"],
                "winner": m["winner"],
                "date": m.get("date") or "",
                "tournament": m.get("tournament") or "",
                "score": m.get("score") or "",
                "round": m.get("round") or "",
            })
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_LIST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} matches to {out_path}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
