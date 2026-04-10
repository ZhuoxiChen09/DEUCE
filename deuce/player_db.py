"""
Player stats database: collect and merge ATP data into a single JSON store.
One record per player; merge updates only non-empty fields so you can build
the DB incrementally from rankings, leaderboards, win/loss index, and TDI.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from deuce.name_utils import normalize_player_name

_PLAYER_NAME_STRIP_RE = re.compile(r"^[+-]?\d+\s*")


def _clean_player_name(name: str) -> str:
    """Remove leading rank movement from name (e.g. '-2Tommy Paul' -> 'Tommy Paul')."""
    if not name:
        return name
    return _PLAYER_NAME_STRIP_RE.sub("", str(name).strip()).strip() or name.strip()


DEFAULT_DB_PATH = "data/player_stats.json"

STAT_KEYS = [
    "player",
    "tour",
    "rank",
    "points",
    "serve_rating",
    "return_rating",
    "pressure_rating",
    "winloss_index",
    "tdi_perf",
    "updated",
    "rankings_updated",
    "leaderboards_updated",
]

RANKINGS_MAP = {"rank": "rank", "official points": "points"}
SERVE_MAP = {"serve_rating": "serve_rating"}
RETURN_MAP = {"return_rating": "return_rating"}
PRESSURE_MAP = {"pressure_rating": "pressure_rating"}
WINLOSS_MAP = {"index": "winloss_index"}
TDI_MAP = {"performance_rating": "tdi_perf"}


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        s = str(value).strip().replace(",", "")
        if not s:
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _find_record(db: List[Dict[str, Any]], name: str) -> Optional[int]:
    """Return index of record whose player name matches (normalized), or None."""
    needle = normalize_player_name(name)
    if not needle:
        return None
    for i, rec in enumerate(db):
        if normalize_player_name(rec.get("player") or "") == needle:
            return i
    return None


def _sanitize_db_player_names(db: List[Dict[str, Any]]) -> None:
    """In-place: strip leading rank movement from every record's player field."""
    for rec in db:
        p = rec.get("player")
        if p and isinstance(p, str):
            cleaned = _clean_player_name(p)
            if cleaned != p:
                rec["player"] = cleaned


def _dedupe_db(db: List[Dict[str, Any]]) -> None:
    """In-place: keep one record per normalized name; merge non-empty fields from duplicates."""
    seen: Dict[str, int] = {}
    to_remove: List[int] = []
    for i, rec in enumerate(db):
        name = (rec.get("player") or "").strip()
        if not name:
            to_remove.append(i)
            continue
        key = normalize_player_name(name)
        if not key:
            to_remove.append(i)
            continue
        if key in seen:
            first_idx = seen[key]
            first = db[first_idx]
            for k, v in rec.items():
                if k == "player":
                    continue
                if v is not None and v != "" and (first.get(k) is None or first.get(k) == ""):
                    first[k] = v
            to_remove.append(i)
        else:
            seen[key] = i
    for i in reversed(to_remove):
        db.pop(i)


def load_db(path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Load player DB from JSON. Returns list of player records; empty list if file missing."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    _sanitize_db_player_names(data)
    _dedupe_db(data)
    return data


def save_db(db: List[Dict[str, Any]], path: str = DEFAULT_DB_PATH) -> None:
    """Save player DB to JSON. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def _merge_rows(
    db: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    field_map: Dict[str, str],
) -> None:
    """Merge new_rows into db in place. field_map: source_key -> db_key."""
    for row in new_rows:
        name = _clean_player_name(row.get("player") or "")
        if not name:
            continue
        idx = _find_record(db, name)
        if idx is None:
            rec: Dict[str, Any] = {
                "player": name,
                "tour": "ATP",
                "updated": datetime.now(tz=timezone.utc).isoformat(),
            }
            for src_key, db_key in field_map.items():
                val = row.get(src_key)
                if val is not None and val != "":
                    f = _safe_float(val)
                    if f is not None:
                        rec[db_key] = f
            db.append(rec)
        else:
            rec = db[idx]
            for src_key, db_key in field_map.items():
                val = row.get(src_key)
                if val is not None and val != "":
                    f = _safe_float(val)
                    if f is not None:
                        rec[db_key] = f
            rec["updated"] = datetime.now(tz=timezone.utc).isoformat()


def update_db_from_atp_datasets(
    path: str = DEFAULT_DB_PATH,
    *,
    atp_rankings: Optional[List[Dict[str, Any]]] = None,
    serve_leaderboard: Optional[List[Dict[str, Any]]] = None,
    return_leaderboard: Optional[List[Dict[str, Any]]] = None,
    pressure_leaderboard: Optional[List[Dict[str, Any]]] = None,
    winloss_index: Optional[List[Dict[str, Any]]] = None,
    tdi_leaderboard: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Load DB, merge in all provided ATP datasets, save. Returns total number of players in DB after merge.
    """
    db = load_db(path)
    if atp_rankings:
        _merge_rows(db, atp_rankings, RANKINGS_MAP)
    if serve_leaderboard:
        _merge_rows(db, serve_leaderboard, SERVE_MAP)
    if return_leaderboard:
        _merge_rows(db, return_leaderboard, RETURN_MAP)
    if pressure_leaderboard:
        _merge_rows(db, pressure_leaderboard, PRESSURE_MAP)
    if winloss_index:
        _merge_rows(db, winloss_index, WINLOSS_MAP)
    if tdi_leaderboard:
        _merge_rows(db, tdi_leaderboard, TDI_MAP)
    save_db(db, path)
    return len(db)


def fix_player_names_and_save(path: str = DEFAULT_DB_PATH) -> int:
    """
    Load DB, strip leading rank movement from player names, dedupe, and save.
    Returns number of records after fix.
    """
    db = load_db(path)
    save_db(db, path)
    return len(db)


def get_player_stats(db: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """Return the record for this player (matched by normalized name), or None."""
    idx = _find_record(db, name)
    if idx is None:
        return None
    return db[idx].copy()


def list_players(db: List[Dict[str, Any]]) -> List[str]:
    """Return list of display names (player field) in the DB."""
    return [r.get("player") or "" for r in db if r.get("player")]
