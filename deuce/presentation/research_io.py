"""Load precomputed walk-forward / research JSON for dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_research_json(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
