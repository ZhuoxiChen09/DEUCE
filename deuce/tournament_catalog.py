"""
Tournament catalog: map tournament names to ATP tier and weight for model training.
Catalog is maintained manually (e.g. data/tournament_catalog.json). Assigns tier
(Grand Slam, Masters 1000, 500, 250, Challenger).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

CatalogEntry = Dict[str, Any]
Catalog = List[CatalogEntry]

TIER_GRAND_SLAM = "grand_slam"
TIER_ATP_FINALS = "atp_finals"
TIER_MASTERS_1000 = "masters_1000"
TIER_ATP_500 = "atp_500"
TIER_ATP_250 = "atp_250"
TIER_CHALLENGER = "challenger"
TIER_UNKNOWN = "unknown"

DEFAULT_WEIGHTS = {
    TIER_GRAND_SLAM: 4.0,
    TIER_ATP_FINALS: 3.5,
    TIER_MASTERS_1000: 3.0,
    TIER_ATP_500: 2.0,
    TIER_ATP_250: 1.0,
    TIER_CHALLENGER: 0.5,
    TIER_UNKNOWN: 1.0,
}

GRAND_SLAM_NAMES = [
    "australian open",
    "roland garros",
    "french open",
    "wimbledon",
    "us open",
]

MASTERS_1000_NAMES = [
    "bnp paribas open",
    "indian wells",
    "miami open",
    "monte-carlo",
    "monte carlo",
    "mutua madrid open",
    "madrid open",
    "internazionali bnl d'italia",
    "internazionali bnl",
    "rome",
    "national bank open",
    "canada open",
    "montreal",
    "toronto",
    "cincinnati open",
    "cincinnati",
    "shanghai masters",
    "rolex shanghai",
    "paris masters",
    "rolex paris masters",
]

ATP_FINALS_NAMES = [
    "nitto atp finals",
    "atp finals",
    "next gen atp finals",
]

ATP_500_NAMES = [
    "dubai duty free",
    "dubai",
    "abierto mexicano",
    "acapulco",
    "barcelona open",
    "barcelona",
    "terra wortmann",
    "halle",
    "queen's",
    "hsbc championships",
    "london",
    "hamburg",
    "bitpanda hamburg",
    "citi dc open",
    "washington",
    "china open",
    "beijing",
    "japan open",
    "tokyo",
    "erste bank open",
    "vienna",
    "swiss indoors",
    "basel",
]

CHALLENGER_PATTERNS = [
    re.compile(r"challenger", re.I),
    re.compile(r"\s+-\s*ch\s*$", re.I),
]

SURFACE_CLAY = "clay"
SURFACE_GRASS = "grass"
SURFACE_HARD_OUTDOOR = "hard_outdoor"
SURFACE_HARD_INDOOR = "hard_indoor"
SURFACE_UNKNOWN = "unknown"

CLAY_NAMES = [
    "clay", "roland garros", "french open", "barcelona open", "madrid open",
    "mutua madrid", "rome", "internazionali bnl", "monte-carlo", "monte carlo",
    "hamburg", "rio open", "buenos aires", "quito", "cordoba", "estoril",
    "marrakech", "munich", "geneva", "lyon", "belgrade", "umag", "gstaad",
    "kitzbuhel", "bastad", "gstaad", "grand prix hassan",
]
GRASS_NAMES = [
    "wimbledon", "halle", "queen's", "stuttgart", "eastbourne", "newport",
    "s-Hertogenbosch", "libema", "mallorca", "grass",
]
INDOOR_NAMES = [
    "indoor", "paris masters", "rolex paris", "vienna", "basel", "swiss indoors",
    "rotterdam", "marseille", "dallas", "montpellier", "sofia", "antwerp",
    "moscow", "stockholm", "atp finals", "nitto atp finals", "next gen",
]

TOURNAMENT_LOCATIONS: Dict[str, str] = {
    "australian open": "Melbourne, Australia",
    "bnp paribas open": "Indian Wells, California",
    "indian wells": "Indian Wells, California",
    "miami open": "Miami, Florida",
    "monte-carlo": "Monte Carlo, Monaco",
    "monte carlo": "Monte Carlo, Monaco",
    "mutua madrid open": "Madrid, Spain",
    "madrid open": "Madrid, Spain",
    "internazionali bnl d'italia": "Rome, Italy",
    "rome": "Rome, Italy",
    "roland garros": "Paris, France",
    "french open": "Paris, France",
    "wimbledon": "London, United Kingdom",
    "national bank open": "Montreal, Canada",
    "cincinnati open": "Cincinnati, Ohio",
    "us open": "New York, USA",
    "shanghai masters": "Shanghai, China",
    "rolex shanghai": "Shanghai, China",
    "paris masters": "Paris, France",
    "rolex paris masters": "Paris, France",
    "dubai": "Dubai, UAE",
    "acapulco": "Acapulco, Mexico",
    "abierto mexicano": "Acapulco, Mexico",
    "barcelona open": "Barcelona, Spain",
    "halle": "Halle, Germany",
    "queen's": "London, United Kingdom",
    "washington": "Washington DC, USA",
    "citi dc open": "Washington DC, USA",
    "beijing": "Beijing, China",
    "china open": "Beijing, China",
    "tokyo": "Tokyo, Japan",
    "japan open": "Tokyo, Japan",
    "vienna": "Vienna, Austria",
    "basel": "Basel, Switzerland",
    "swiss indoors": "Basel, Switzerland",
    "nitto atp finals": "Turin, Italy",
    "atp finals": "Turin, Italy",
}

DEFAULT_CATALOG_PATH = "data/tournament_catalog.json"


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def _match_surface(name: str) -> str:
    n = _normalize_name(name)
    if not n:
        return SURFACE_UNKNOWN
    for s in CLAY_NAMES:
        if s in n:
            return SURFACE_CLAY
    for s in GRASS_NAMES:
        if s in n:
            return SURFACE_GRASS
    for s in INDOOR_NAMES:
        if s in n:
            return SURFACE_HARD_INDOOR
    return SURFACE_HARD_OUTDOOR


def get_surface(tournament_name: str) -> str:
    return _match_surface(tournament_name)


def _month_from_date(date_str: str) -> int:
    if not date_str or len(date_str) < 6:
        return 0
    try:
        return int(date_str[4:6])
    except (ValueError, TypeError):
        return 0


def _match_tier(name: str) -> str:
    n = _normalize_name(name)
    if not n:
        return TIER_UNKNOWN
    for g in GRAND_SLAM_NAMES:
        if g in n:
            return TIER_GRAND_SLAM
    for m in ATP_FINALS_NAMES:
        if m in n:
            return TIER_ATP_FINALS
    for m in MASTERS_1000_NAMES:
        if m in n:
            return TIER_MASTERS_1000
    for m in ATP_500_NAMES:
        if m in n:
            return TIER_ATP_500
    for pat in CHALLENGER_PATTERNS:
        if pat.search(n):
            return TIER_CHALLENGER
    return TIER_ATP_250


def get_tier(tournament_name: str) -> str:
    return _match_tier(tournament_name)


def get_weight(tournament_name: str, weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    tier = _match_tier(tournament_name)
    return float(w.get(tier, w.get(TIER_UNKNOWN, 1.0)))


def load_catalog(path: str = DEFAULT_CATALOG_PATH) -> List[Dict[str, Any]]:
    out = load_catalog_full(path)
    return out.get("tournaments", [])


def load_catalog_full(path: str = DEFAULT_CATALOG_PATH) -> Dict[str, Any]:
    p = Path(path)
    default = {"tournaments": [], "round_importance_map": {}}
    if not p.exists():
        return default
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return default
    if isinstance(data, list):
        return {"tournaments": data, "round_importance_map": {}}
    if isinstance(data, dict):
        tournaments = data.get("tournaments", data.get("tournament_catalog", []))
        rom = data.get("round_importance_map") or {}
        return {"tournaments": tournaments, "round_importance_map": rom}
    return default


def save_catalog(catalog: List[Dict[str, Any]], path: str = DEFAULT_CATALOG_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"tournaments": catalog}, f, indent=2)


def get_location_for_tournament(tournament_name: str) -> str:
    n = _normalize_name(tournament_name)
    for key, loc in TOURNAMENT_LOCATIONS.items():
        if key in n or n in key:
            return loc
    return ""


def enrich_catalog_with_weather(
    catalog: Catalog,
    *,
    use_web_search: bool = True,
    limit: Optional[int] = None,
) -> None:
    if not use_web_search:
        return
    try:
        from weather_enrichment import enrich_entry_weather
    except ImportError:
        return
    n = 0
    for entry in catalog:
        if limit is not None and n >= limit:
            break
        if entry.get("temperature_avg") is not None and entry.get("humidity_avg") is not None:
            continue
        location = (entry.get("location") or "").strip()
        if not location:
            location = get_location_for_tournament(entry.get("name") or "")
        month = entry.get("month") or _month_from_date(entry.get("date_start") or "")
        if not location or not month:
            continue
        enrich_entry_weather(entry, location, month)
        n += 1


def get_tournament_context(
    tournament_name: str,
    catalog: Optional[Catalog] = None,
) -> Dict[str, Any]:
    ctx = {
        "surface": SURFACE_UNKNOWN,
        "month": 0,
        "temperature_avg": None,
        "humidity_avg": None,
        "surface_speed_rating": None,
        "court_type": None,
        "altitude_m": None,
        "event_id": None,
        "tier": None,
        "country": None,
    }
    if catalog:
        n = _normalize_name(tournament_name)
        for entry in catalog:
            if _normalize_name(entry.get("name") or "") == n:
                surface = entry.get("surface") or _match_surface(tournament_name)
                ctx["surface"] = surface
                ctx["month"] = entry.get("month") or _month_from_date(entry.get("date_start") or "")
                ctx["temperature_avg"] = entry.get("temperature_avg")
                ctx["humidity_avg"] = entry.get("humidity_avg")
                ctx["surface_speed_rating"] = entry.get("surface_speed_rating")
                ctx["court_type"] = entry.get("court_type")
                ctx["altitude_m"] = entry.get("altitude_m")
                ctx["weight"] = entry.get("weight")
                ctx["event_id"] = entry.get("event_id")
                ctx["tier"] = entry.get("tier")
                loc = (entry.get("location") or "").strip()
                if loc and "," in loc:
                    ctx["country"] = loc.split(",")[-1].strip()[:3].upper()
                elif loc:
                    ctx["country"] = loc[:3].upper()
                return ctx
    ctx["surface"] = _match_surface(tournament_name)
    return ctx


def get_weight_for_match(
    tournament_name: str,
    catalog: Optional[Catalog] = None,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    if catalog:
        n = _normalize_name(tournament_name)
        for entry in catalog:
            if _normalize_name(entry.get("name") or "") == n:
                return float(entry.get("weight", 1.0))
    return get_weight(tournament_name, weights=weights)
