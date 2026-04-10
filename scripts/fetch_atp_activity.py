"""
Fetch ATP player activity for one or more players and write a match list CSV with
player_a, player_b, winner, date, tournament, score, round for build_historical_from_db.py.

Deduplicates matches that appear on both players' activity pages.
Requires Playwright: pip install playwright && playwright install chromium

Usage:
  python -m scripts.fetch_atp_activity carlos-alcaraz/a0e2 --year 2026
  python -m scripts.fetch_atp_activity carlos-alcaraz/a0e2 arthur-rinderknech/rc91 --year all --out data/match_list.csv
"""
import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from atp_activity import fetch_player_activity

MATCH_LIST_FIELDS = ["player_a", "player_b", "winner", "date", "tournament", "score", "round"]


def _dedupe_matches(matches: list) -> list:
    """One row per unique pair (player_a, player_b); keep first occurrence with all fields."""
    seen = {}
    for m in matches:
        p1, p2 = m["player_a"], m["player_b"]
        winner_name = p1 if m["winner"] == "A" else p2
        first, second = (p1, p2) if p1 <= p2 else (p2, p1)
        key = (first, second)
        if key not in seen:
            seen[key] = {
                "player_a": first,
                "player_b": second,
                "winner": "A" if winner_name == first else "B",
                "date": m.get("year") or m.get("date") or "",
                "tournament": m.get("tournament") or "",
                "score": m.get("score") or "",
                "round": m.get("round") or "",
            }
    return list(seen.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch ATP player activity and write match list CSV."
    )
    parser.add_argument(
        "slugs",
        nargs="+",
        help="Player slugs, e.g. carlos-alcaraz/a0e2 arthur-rinderknech/rc91",
    )
    parser.add_argument(
        "--year",
        default="all",
        help="Year filter: all, 2026, 2025, ...",
    )
    parser.add_argument(
        "--out",
        default=str(_REPO_ROOT / "data" / "match_list.csv"),
        help="Output CSV path (default: data/match_list.csv).",
    )
    args = parser.parse_args()

    all_matches = []
    for slug in args.slugs:
        slug = slug.strip()
        if not slug:
            continue
        print(f"Fetching {slug} (year={args.year})...", file=sys.stderr)
        try:
            matches = fetch_player_activity(slug, year=args.year)
            all_matches.extend(matches)
            print(f"  {len(matches)} matches.", file=sys.stderr)
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            return 1

    rows = _dedupe_matches(all_matches)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MATCH_LIST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} unique matches to {out_path}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
