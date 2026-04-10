"""
Enrich tournament catalog with typical temperature and humidity for the location and month
when the tournament is played. Uses web search (duckduckgo-search) to find climate data.
"""
import re
from typing import Any, Dict, Optional, Tuple

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _search_weather(location: str, month: int) -> str:
    """Run web search for average temperature and humidity; return snippet text."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return ""
    if not location or month < 1 or month > 12:
        return ""
    month_name = MONTH_NAMES[month] if month < len(MONTH_NAMES) else ""
    query = f"average temperature humidity {location} {month_name}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return ""
        return " ".join((r.get("body") or "") for r in results)
    except Exception:
        return ""


def _parse_temperature_from_text(text: str) -> Optional[float]:
    """Parse first reasonable temperature in °F or °C from text; return Fahrenheit."""
    if not text:
        return None
    # Prefer °F: "72°F", "72 °F", "72F"
    m = re.search(r"(\d{1,3})\s*°?\s*F\b", text, re.I)
    if m:
        t = float(m.group(1))
        if 30 <= t <= 120:
            return t
    # °C: convert to F
    m = re.search(r"(\d{1,3})\s*°?\s*C\b", text, re.I)
    if m:
        c = float(m.group(1))
        if -10 <= c <= 50:
            return c * 9 / 5 + 32
    # "high 75" or "75 degrees"
    m = re.search(r"(?:high|avg|average|mean)\s+(\d{1,3})\b", text, re.I)
    if m:
        t = float(m.group(1))
        if 30 <= t <= 120:
            return t
    return None


def _parse_humidity_from_text(text: str) -> Optional[float]:
    """Parse first reasonable humidity percentage from text."""
    if not text:
        return None
    m = re.search(r"(\d{1,3})\s*%\s*(?:humidity|humid)", text, re.I)
    if m:
        h = float(m.group(1))
        if 0 <= h <= 100:
            return h
    m = re.search(r"humidity\s*(?:is|:)?\s*(\d{1,3})\s*%?", text, re.I)
    if m:
        h = float(m.group(1))
        if 0 <= h <= 100:
            return h
    m = re.search(r"(\d{1,3})\s*%\s*(?:relative)?", text, re.I)
    if m:
        h = float(m.group(1))
        if 0 <= h <= 100:
            return h
    return None


def get_weather_for_place_month(
    location: str,
    month: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (temperature_avg_f, humidity_avg_pct) for the place and month.
    Uses web search; returns (None, None) if unavailable or parsing fails.
    """
    if not location or month < 1 or month > 12:
        return None, None
    text = _search_weather(location, month)
    temp = _parse_temperature_from_text(text)
    humidity = _parse_humidity_from_text(text)
    return temp, humidity


def enrich_entry_weather(
    entry: Dict[str, Any],
    location: str,
    month: int,
) -> None:
    """Update entry with temperature_avg and humidity_avg in place."""
    temp, humidity = get_weather_for_place_month(location, month)
    if temp is not None:
        entry["temperature_avg"] = round(temp, 1)
    if humidity is not None:
        entry["humidity_avg"] = round(humidity, 1)
