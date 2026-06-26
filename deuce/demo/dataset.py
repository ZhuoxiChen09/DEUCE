"""Build the lightweight historical dataset used by the portfolio demo."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEMO_START_YEAR = 2018
DEMO_END_YEAR = 2025
DEMO_SURFACES = ("Hard", "Clay", "Grass")
DEMO_LEVELS = ("G", "M", "500", "250", "F", "A", "O")
DEFAULT_MAX_PER_GROUP = 12
DEMO_DATASET_NAME = "demo_matches.csv"


def _stable_key(row: dict[str, str]) -> str:
    parts = [
        row.get("date", ""),
        row.get("tournament", ""),
        row.get("round", ""),
        row.get("player_a", ""),
        row.get("player_b", ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _valid_demo_row(row: dict[str, str]) -> bool:
    date = (row.get("date") or "").strip()
    if len(date) < 4 or not date[:4].isdigit():
        return False
    year = int(date[:4])
    if year < DEMO_START_YEAR or year > DEMO_END_YEAR:
        return False
    if (row.get("surface") or "").strip() not in DEMO_SURFACES:
        return False
    if (row.get("tourney_level") or "").strip() not in DEMO_LEVELS:
        return False
    for key in ("rank_a", "rank_b", "points_a", "points_b", "player_a", "player_b", "winner"):
        if not (row.get(key) or "").strip():
            return False
    return (row.get("winner") or "").strip().upper() in {"A", "B"}


def build_demo_dataset(
    source_csv: Path | str,
    output_csv: Path | str,
    *,
    max_per_group: int = DEFAULT_MAX_PER_GROUP,
) -> dict[str, Any]:
    """
    Create a deterministic compact slice of the existing historical training CSV.

    Rows are grouped by year, surface, and tournament level. Within each group,
    a stable SHA-256 ordering selects up to ``max_per_group`` rows so the output
    is reproducible across machines without relying on Python's randomized hash.
    """
    src = Path(source_csv)
    out = Path(output_csv)
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if not _valid_demo_row(row):
                continue
            key = (
                (row.get("date") or "")[:4],
                (row.get("surface") or "").strip(),
                (row.get("tourney_level") or "").strip(),
            )
            groups[key].append(row)

    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=_stable_key)
        selected.extend(rows[:max_per_group])

    selected.sort(
        key=lambda r: (
            r.get("date", ""),
            r.get("tournament", ""),
            r.get("round", ""),
            r.get("player_a", ""),
            r.get("player_b", ""),
        )
    )
    if not selected:
        raise ValueError("Demo dataset filter selected zero rows.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    return {
        "source_csv": str(src),
        "output_csv": str(out),
        **_summary_from_rows(selected, fieldnames),
        "filter": {
            "years": [DEMO_START_YEAR, DEMO_END_YEAR],
            "surfaces": list(DEMO_SURFACES),
            "levels": list(DEMO_LEVELS),
            "max_per_year_surface_level": max_per_group,
            "required_fields": ["rank_a", "rank_b", "points_a", "points_b"],
        },
    }


def _summary_from_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, Any]:
    dates = [row["date"] for row in rows if row.get("date")]
    surface_counts = Counter(row.get("surface", "") for row in rows)
    level_counts = Counter(row.get("tourney_level", "") for row in rows)
    year_counts = Counter((row.get("date", "")[:4]) for row in rows)
    return {
        "row_count": len(rows),
        "column_count": len(fieldnames),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "surfaces": dict(sorted(surface_counts.items())),
        "levels": dict(sorted(level_counts.items())),
        "years": dict(sorted(year_counts.items())),
    }


def summarize_demo_dataset(path: Path | str) -> dict[str, Any]:
    """Summarize a committed demo CSV without modifying it."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Demo CSV not found: {p}")
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return {
        "source_csv": str(p),
        "output_csv": str(p),
        **_summary_from_rows(rows, fieldnames),
        "filter": {
            "years": [DEMO_START_YEAR, DEMO_END_YEAR],
            "surfaces": list(DEMO_SURFACES),
            "levels": list(DEMO_LEVELS),
            "max_per_year_surface_level": DEFAULT_MAX_PER_GROUP,
            "required_fields": ["rank_a", "rank_b", "points_a", "points_b"],
        },
    }
