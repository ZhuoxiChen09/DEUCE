"""
Kalshi public (unauthenticated) API client.
Series, events, markets, and orderbook data.
"""
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_TIMEOUT_S = 20
RETRIES = 2
RETRY_BACKOFF_S = 1.0

# Kalshi markets UI (replace {ticker} with series_ticker e.g. KXWTAMATCH)
MARKETS_UI_BASE = "https://kalshi.com/markets"


def safe_text(value: Any) -> str:
    """Normalize value for display; avoid Windows console encoding errors."""
    if value is None:
        return ""
    s = str(value)
    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _raise_for_kalshi_errors(payload: Dict[str, Any]) -> None:
    if isinstance(payload, dict) and "error" in payload and payload["error"]:
        raise RuntimeError(str(payload["error"]))


def get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    retries: int = RETRIES,
    backoff_s: float = RETRY_BACKOFF_S,
) -> Dict[str, Any]:
    """GET JSON from URL with optional retries on 5xx/timeout."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                _raise_for_kalshi_errors(payload)
            return payload
        except (requests.RequestException, RuntimeError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_s)
    raise last_exc  # type: ignore[misc]


def get_series(series_ticker: str) -> Dict[str, Any]:
    return get_json(f"{BASE_URL}/series/{series_ticker}")


def list_series(
    *,
    limit: int,
    max_pages: int,
) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(max_pages):
        params: Dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        payload = get_json(f"{BASE_URL}/series", params=params)
        batch = payload.get("series") or []
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected series payload shape: {payload}")
        series.extend(batch)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return series


def get_markets(
    *,
    series_ticker: str,
    status: str,
    limit: int,
    max_pages: int,
) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(max_pages):
        params: Dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor
        payload = get_json(f"{BASE_URL}/markets", params=params)
        batch = payload.get("markets") or []
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected markets payload shape: {payload}")
        markets.extend(batch)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return markets


def list_markets(
    *,
    status: str,
    limit: int,
    max_pages: int,
) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    for _ in range(max_pages):
        params: Dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        payload = get_json(f"{BASE_URL}/markets", params=params)
        batch = payload.get("markets") or []
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected markets payload shape: {payload}")
        markets.extend(batch)
        cursor = payload.get("cursor")
        if not cursor:
            break
    return markets


def get_event(event_ticker: str) -> Dict[str, Any]:
    return get_json(f"{BASE_URL}/events/{event_ticker}")


def get_orderbook(market_ticker: str) -> Dict[str, Any]:
    return get_json(f"{BASE_URL}/markets/{market_ticker}/orderbook")


# --- Parsing helpers ---

_PLAYER_FROM_TITLE_RE = re.compile(r"^Will\s+(?P<player>.+?)\s+win\b", re.IGNORECASE)
_ROUND_FROM_TITLE_RE = re.compile(r":\s*(?P<round>[^:?]+?)\s+match\?", re.IGNORECASE)
_ROUND_FROM_RULES_RE = re.compile(
    r"\b("
    r"Round Of 128|Round Of 64|Round Of 32|Round Of 16|"
    r"Quarterfinals?|Semifinals?|Final|"
    r"1st Round|2nd Round|3rd Round|4th Round"
    r")\b",
    re.IGNORECASE,
)


def parse_iso8601_utc(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def format_iso8601_local(ts: Any) -> str:
    """
    Format an ISO timestamp in the local system timezone.

    Keeps the source UTC value separate from UI display so internal logic can still
    operate on normalized timestamps.
    """
    dt = parse_iso8601_utc(ts)
    if dt is None:
        return ""
    local_dt = dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %I:%M %p %Z")


def infer_market_start_state(
    expected_expiration_time: Any,
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Infer whether a market is still pregame based on Kalshi's expected expiration time.

    For tennis winner markets, this timestamp is a practical proxy for scheduled match start.
    We keep the label explicitly inferential because delays are possible.
    """
    expected_dt = parse_iso8601_utc(expected_expiration_time)
    if expected_dt is None:
        return {
            "scheduled_start": None,
            "is_started": None,
            "phase": "unknown",
            "phase_label": "Start unknown",
            "phase_detail": "No Kalshi scheduled start available.",
            "inference_source": "kalshi_expected_expiration_time",
        }
    now_utc = now.astimezone(timezone.utc) if now is not None else datetime.now(timezone.utc)
    is_started = now_utc >= expected_dt
    return {
        "scheduled_start": expected_dt.isoformat().replace("+00:00", "Z"),
        "scheduled_start_local": format_iso8601_local(expected_dt.isoformat().replace("+00:00", "Z")),
        "is_started": is_started,
        "phase": "started_or_live_like" if is_started else "pregame",
        "phase_label": "Started / live-like" if is_started else "Pregame",
        "phase_detail": (
            "Scheduled start has passed per Kalshi timing."
            if is_started
            else "Scheduled start has not passed yet."
        ),
        "inference_source": "kalshi_expected_expiration_time",
    }


def extract_tennis_round(*texts: Any) -> str:
    """Best-effort extraction of round text from Kalshi tennis market titles/rules."""
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        match = _ROUND_FROM_TITLE_RE.search(text)
        if match:
            return match.group("round").strip()
        match = _ROUND_FROM_RULES_RE.search(text)
        if match:
            return match.group(1).strip()
    return ""


def summarize_tennis_event(
    event_payload: Dict[str, Any],
    *,
    fallback_expected_expiration_time: Any = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Normalize Kalshi tennis event payload into UI-friendly metadata.
    """
    event = event_payload.get("event", {}) if isinstance(event_payload, dict) else {}
    markets = event_payload.get("markets", []) if isinstance(event_payload, dict) else []
    first_market = markets[0] if isinstance(markets, list) and markets else {}
    product_meta = event.get("product_metadata", {}) if isinstance(event, dict) else {}

    expected_expiration_time = (
        first_market.get("expected_expiration_time")
        if isinstance(first_market, dict) and first_market.get("expected_expiration_time")
        else fallback_expected_expiration_time
    )
    start_state = infer_market_start_state(expected_expiration_time, now=now)
    round_name = extract_tennis_round(
        first_market.get("title") if isinstance(first_market, dict) else None,
        first_market.get("rules_primary") if isinstance(first_market, dict) else None,
        first_market.get("rules_secondary") if isinstance(first_market, dict) else None,
    )
    competition = ""
    if isinstance(product_meta, dict):
        competition = str(product_meta.get("competition") or "").strip()
    info_line = " · ".join(part for part in [competition, round_name] if part)

    return {
        "event_title": str(event.get("title") or "").strip(),
        "event_subtitle": str(event.get("sub_title") or "").strip(),
        "event_category": str(event.get("category") or "").strip(),
        "event_series_ticker": str(event.get("series_ticker") or "").strip(),
        "event_competition": competition,
        "event_scope": str(product_meta.get("competition_scope") or "").strip() if isinstance(product_meta, dict) else "",
        "event_round": round_name,
        "event_info_line": info_line,
        "scheduled_start": start_state["scheduled_start"],
        "scheduled_start_local": start_state.get("scheduled_start_local", ""),
        "is_started": start_state["is_started"],
        "market_phase": start_state["phase"],
        "market_phase_label": start_state["phase_label"],
        "market_phase_detail": start_state["phase_detail"],
        "market_phase_source": start_state["inference_source"],
    }


def player_from_market_title(market: Dict[str, Any]) -> Optional[str]:
    title = market.get("title")
    if not isinstance(title, str):
        return None
    m = _PLAYER_FROM_TITLE_RE.match(title.strip())
    if not m:
        return None
    return m.group("player").strip()


def to_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None
    return None


def matches_query(market: Dict[str, Any], query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    fields = [
        market.get("title"),
        market.get("ticker"),
        market.get("category"),
        market.get("subcategory"),
        market.get("series_ticker"),
        market.get("event_ticker"),
    ]
    haystack = " | ".join(str(v).lower() for v in fields if v is not None)
    return q in haystack


def series_matches_query(series: Dict[str, Any], query: str) -> bool:
    q = query.strip().lower()
    if not q:
        return False
    fields = [
        series.get("title"),
        series.get("ticker"),
        series.get("category"),
        series.get("subcategory"),
        series.get("frequency"),
    ]
    haystack = " | ".join(str(v).lower() for v in fields if v is not None)
    return q in haystack


# Keywords that identify tennis series (WTA, ATP, Challenger, 125K, etc.)
TENNIS_SERIES_KEYWORDS = ("tennis", "wta", "atp", "challenger", "125k", "indian wells")


def series_matches_tennis(series: Dict[str, Any]) -> bool:
    """True if series looks like any tennis circuit (WTA, ATP, Challenger, 125K, etc.)."""
    if not isinstance(series, dict):
        return False
    fields = [
        series.get("title"),
        series.get("ticker"),
        series.get("category"),
        series.get("subcategory"),
    ]
    haystack = " | ".join(str(v).lower() for v in fields if v is not None)
    return any(kw in haystack for kw in TENNIS_SERIES_KEYWORDS)


def top_orderbook_levels(levels: Any, n: int) -> List[Tuple[float, float]]:
    if not isinstance(levels, list):
        return []
    out: List[Tuple[float, float]] = []
    for item in levels[:n]:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ):
            out.append((float(item[0]), float(item[1])))
    return out
