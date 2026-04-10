"""
CLI for Kalshi public market data: series mode or search mode (e.g. tennis matchups),
and ATP singles rankings scraping.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import re
import requests
from bs4 import BeautifulSoup

from api import (
    BASE_URL,
    MARKETS_UI_BASE,
    extract_tennis_round,
    format_iso8601_local,
    get_event,
    get_markets,
    get_orderbook,
    infer_market_start_state,
    get_series,
    list_markets,
    list_series,
    matches_query,
    parse_iso8601_utc,
    player_from_market_title,
    safe_text,
    series_matches_query,
    series_matches_tennis,
    summarize_tennis_event,
    to_float,
    top_orderbook_levels,
)

# Optional prediction layer (model_*); import only when enriching matchups
try:
    from model_features import build_player_stats_from_datasets
    from model_predict import (
        get_default_core_path,
        get_default_enhanced_path,
        get_default_model_path,
        load_model,
        predict_match_probability,
    )
    from evaluation import evaluate as evaluate_prediction
    _prediction_available = True
except ImportError:
    _prediction_available = False
    evaluate_prediction = None

try:
    from deuce.player_db import DEFAULT_DB_PATH as PLAYER_DB_DEFAULT_PATH, update_db_from_atp_datasets
    _player_db_available = True
except ImportError:
    _player_db_available = False
    PLAYER_DB_DEFAULT_PATH = "data/player_stats.json"

try:
    from deuce.market.pricing import TwoSidedQuote
except ImportError:
    TwoSidedQuote = None  # type: ignore


SERIES_DELAY_S = 0.12  # Delay between series requests (reduced for faster search)
MAX_SERIES_WORKERS = 4  # Parallel get_markets across series
ATP_SINGLES_RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=0-5000"
ATP_LEADERBOARD_URL = (
    "https://www.atptour.com/en/stats/leaderboard?boardType={board_type}&timeFrame=52week"
    "&surface={surface}&versusRank=all&formerNo1=false"
)
ATP_SERVE_LEADERBOARD_URL = (
    "https://www.atptour.com/en/stats/leaderboard?boardType=serve&timeFrame=52week"
    "&surface={surface}&versusRank=all&formerNo1=false"
)
ATP_SERVE_SURFACES = ("all", "Clay", "Grass", "Hard")
ATP_LEADERBOARD_TYPES = ("serve", "return", "pressure")

# Win/Loss Index: https://www.atptour.com/en/stats/win-loss-index
ATP_WINLOSS_INDEX_URL = (
    "https://www.atptour.com/en/stats/win-loss-index"
    "?indexCategory={index_category}&timeCategory={time_category}&country=all&sortBy=index&sortDirection=desc"
)
# (url slug, display label) for indexCategory. Slugs verified/inferred from ATP site.
ATP_WINLOSS_INDEX_CATEGORIES = [
    ("all", "Overall"),
    ("after1stsetloss", "After Losing First Set"),
    ("after1stsetwin", "After Winning First Set"),
    ("decidingset", "Deciding Set"),
    ("5thsetrecord", "5th Set Record"),
    ("finals", "Finals"),
    ("grandslam", "Grand Slams"),
    ("indoor", "Indoor"),
    ("lefthanded", "Left Handed"),
    ("masters1000", "Masters 1000"),
    ("carpet", "On Carpet"),
    ("clay", "On Clay"),
    ("grass", "On Grass"),
    ("hard", "On Hard"),
    ("outdoor", "Outdoor"),
    ("righthanded", "Right Handed"),
    ("tiebreak", "Tie-Break"),
    ("versustopten", "Versus Top Ten"),
]
ATP_WINLOSS_TIME_CATEGORIES = [
    ("roll", "52 Week"),
    ("career", "Career"),
    ("ytd", "Year To Date"),
]
ATP_WINLOSS_INDEX_SLUGS = [slug for slug, _ in ATP_WINLOSS_INDEX_CATEGORIES]
ATP_WINLOSS_TIME_SLUGS = [slug for slug, _ in ATP_WINLOSS_TIME_CATEGORIES]

# TDI Insights Leaderboard: https://www.atptour.com/en/stats/tdi-leaderboard
ATP_TDI_LEADERBOARD_URL = (
    "https://www.atptour.com/en/stats/tdi-leaderboard"
    "?timeFrame={time_frame}&versusRank=all&surface={surface}&sortBy=Performance&sortOrder=dsc"
)
ATP_TDI_TIME_FRAMES = [
    ("average52Week", "Last 52 Weeks"),
    ("2025", "2025"),
]
ATP_TDI_TIME_SLUGS = [slug for slug, _ in ATP_TDI_TIME_FRAMES]


def _print_safe(s: str) -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _fetch_atp_rankings(limit: int) -> List[Dict[str, str]]:
    """
    Fetch ATP singles rankings table (top `limit` rows) from the public rankings page.
    Returns a list of dicts with keys:
    Rank, Player, Age, Official Points, +/-, Tourn Played, Dropping, Next Best.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    resp = requests.get(ATP_SINGLES_RANKINGS_URL, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # Use the full table that includes Age (second table on page). The first table has
    # only Rank, Player, Official Points, Next Best.
    tables = soup.find_all("table")
    table = None
    for t in tables:
        ths = [th.get_text(strip=True).lower() for th in t.find_all("th")]
        if "age" in ths and "official points" in " ".join(ths):
            table = t
            break
    if table is None:
        table = soup.find("table")  # fallback to first table

    if table is None:
        return []

    # Full table columns: Rank, Player, Age, Official Points, +/-, Tourn Played, Dropping, Next Best
    # Body rows have 8 tds (no hidden header cell in tbody).
    rows: List[Dict[str, str]] = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        texts = [td.get_text(strip=True) for td in tds]
        # If 8 columns: rank, player, age, official points, +/-, tourn played, dropping, next best
        # If 4 columns (short table): rank, player, official points, next best -> no age
        if len(texts) >= 8:
            row = {
                "rank": texts[0],
                "player": texts[1],
                "age": texts[2],
                "official points": texts[3],
                "+/-": texts[4],
                "tourn played": texts[5],
                "dropping": texts[6],
                "next best": texts[7] if len(texts) > 7 else "",
            }
        else:
            # Short table: rank, player, official points, next best (4 cols)
            row = {
                "rank": texts[0] if len(texts) > 0 else "",
                "player": texts[1] if len(texts) > 1 else "",
                "age": "",
                "official points": texts[2] if len(texts) > 2 else "",
                "+/-": "",
                "tourn played": "",
                "dropping": "",
                "next best": texts[3] if len(texts) > 3 else "",
            }
        if (row.get("player") or "").strip() or (row.get("official points") or "").strip():
            rows.append(row)
        if 0 < limit <= len(rows):
            break

    return rows


def _normalize_atp_rows(rows: List[Dict[str, str]]) -> None:
    """In-place: clean rank, split movement from player name, set pts chg."""
    for idx, r in enumerate(rows, start=1):
        rank_text = r.get("rank", "") or ""
        m = re.search(r"\d+", rank_text)
        if m:
            r["rank"] = m.group(0)
        else:
            r["rank"] = str(idx)
    for r in rows:
        text = r.get("player", "") or ""
        m = re.match(r"^([+-]?\d+)(.+)$", text)
        rank_movement = ""
        name = text
        if m:
            rank_movement, name = m.groups()
            rank_movement = rank_movement.strip()
            name = name.strip()
        r["movement"] = rank_movement
        r["player"] = name
        r["pts chg"] = (r.get("+/-") or "").strip()


def get_atp_rankings_display(limit: int = 25) -> List[Dict[str, str]]:
    """
    Fetch ATP singles rankings and return rows ready for display (rank, player, movement,
    age, official points, pts chg, tourn played). For use by CLI or UI.
    """
    rows = _fetch_atp_rankings(limit)
    if rows:
        _normalize_atp_rows(rows)
    return rows


def _print_atp_rankings_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        _print_safe("No ATP rankings rows parsed from the rankings page.")
        return
    _normalize_atp_rows(rows)

    headers = ["Rank", "Player", "Movement", "Age", "Official Points", "Pts Chg", "Tourn Played"]

    def _col_width(key: str, header: str) -> int:
        max_cell = max(len(r.get(key, "")) for r in rows)
        return max(max_cell, len(header))

    widths = {
        "Rank": _col_width("rank", "Rank"),
        "Player": _col_width("player", "Player"),
        "Movement": _col_width("movement", "Movement"),
        "Age": _col_width("age", "Age"),
        "Official Points": _col_width("official points", "Official Points"),
        "Pts Chg": _col_width("pts chg", "Pts Chg"),
        "Tourn Played": _col_width("tourn played", "Tourn Played"),
    }

    def _fmt_row(r: Dict[str, str]) -> str:
        return (
            f"{r.get('rank', '').rjust(widths['Rank'])}  "
            f"{r.get('player', '').ljust(widths['Player'])}  "
            f"{r.get('movement', '').rjust(widths['Movement'])}  "
            f"{r.get('age', '').rjust(widths['Age'])}  "
            f"{r.get('official points', '').rjust(widths['Official Points'])}  "
            f"{r.get('pts chg', '').rjust(widths['Pts Chg'])}  "
            f"{r.get('tourn played', '').rjust(widths['Tourn Played'])}"
        )

    header_line = (
        f"{'Rank'.rjust(widths['Rank'])}  "
        f"{'Player'.ljust(widths['Player'])}  "
        f"{'Movement'.rjust(widths['Movement'])}  "
        f"{'Age'.rjust(widths['Age'])}  "
        f"{'Official Points'.rjust(widths['Official Points'])}  "
        f"{'Pts Chg'.rjust(widths['Pts Chg'])}  "
        f"{'Tourn Played'.rjust(widths['Tourn Played'])}"
    )
    sep_line = "-" * len(header_line)

    _print_safe(header_line)
    _print_safe(sep_line)
    for r in rows:
        _print_safe(_fmt_row(r))


def _fetch_atp_leaderboard(
    board_type: str,
    surface: str,
    limit: int,
) -> List[Dict[str, str]]:
    """
    Fetch ATP leaderboard (52-week) for the given board_type and surface using Playwright.
    board_type: one of 'serve', 'return', 'pressure'.
    surface: one of 'all', 'Clay', 'Grass', 'Hard'.
    Returns list of dicts with board-specific keys.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required for ATP leaderboards. "
            "Install with: pip install playwright && playwright install chromium"
        ) from None

    board_type = (board_type or "serve").strip().lower()
    if board_type not in ATP_LEADERBOARD_TYPES:
        board_type = "serve"
    surface = surface.strip() if surface else "all"
    if surface not in ATP_SERVE_SURFACES:
        surface = "all"
    url = ATP_LEADERBOARD_URL.format(board_type=board_type, surface=surface)

    rows: List[Dict[str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=20000)
            page.wait_for_selector("table tbody tr", state="visible", timeout=15000)
            page.wait_for_timeout(1500)
            html = page.content()
        finally:
            browser.close()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return rows

    tbody = table.find("tbody")
    if not tbody:
        return rows

    def _is_rank_col(t: str) -> bool:
        return bool(t and (t.isdigit() or t.rstrip() and t.rstrip()[-1].isdigit()))

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        texts = [td.get_text(strip=True) for td in tds]
        if len(texts) < 3 or not _is_rank_col(texts[0]):
            continue
        if board_type == "serve":
            row = {
                "rank": texts[0] if len(texts) > 0 else "",
                "player": texts[1] if len(texts) > 1 else "",
                "serve_rating": texts[2] if len(texts) > 2 else "",
                "pct_1st_serve": texts[3] if len(texts) > 3 else "",
                "pct_1st_serve_pts_won": texts[4] if len(texts) > 4 else "",
                "pct_2nd_serve_pts_won": texts[5] if len(texts) > 5 else "",
                "pct_service_games_won": texts[6] if len(texts) > 6 else "",
                "avg_aces_per_match": texts[7] if len(texts) > 7 else "",
                "avg_double_faults_per_match": texts[8] if len(texts) > 8 else "",
            }
        elif board_type == "return":
            row = {
                "rank": texts[0] if len(texts) > 0 else "",
                "player": texts[1] if len(texts) > 1 else "",
                "return_rating": texts[2] if len(texts) > 2 else "",
                "pct_1st_serve_return_pts_won": texts[3] if len(texts) > 3 else "",
                "pct_2nd_serve_return_pts_won": texts[4] if len(texts) > 4 else "",
                "pct_return_games_won": texts[5] if len(texts) > 5 else "",
                "pct_break_pts_converted": texts[6] if len(texts) > 6 else "",
            }
        else:  # pressure
            row = {
                "rank": texts[0] if len(texts) > 0 else "",
                "player": texts[1] if len(texts) > 1 else "",
                "pressure_rating": texts[2] if len(texts) > 2 else "",
                "pct_break_pts_converted": texts[3] if len(texts) > 3 else "",
                "pct_break_pts_saved": texts[4] if len(texts) > 4 else "",
                "pct_tiebreaks_won": texts[5] if len(texts) > 5 else "",
                "pct_deciding_sets_won": texts[6] if len(texts) > 6 else "",
            }
        if (row.get("player") or "").strip():
            rows.append(row)
        if 0 < limit <= len(rows):
            break

    return rows


def _fetch_atp_serve_leaderboard(surface: str, limit: int) -> List[Dict[str, str]]:
    """Fetch ATP Serve leaderboard (52-week)."""
    return _fetch_atp_leaderboard("serve", surface, limit)


def _fetch_atp_return_leaderboard(surface: str, limit: int) -> List[Dict[str, str]]:
    """Fetch ATP Return leaderboard (52-week)."""
    return _fetch_atp_leaderboard("return", surface, limit)


def _fetch_atp_pressure_leaderboard(surface: str, limit: int) -> List[Dict[str, str]]:
    """Fetch ATP Under Pressure leaderboard (52-week)."""
    return _fetch_atp_leaderboard("pressure", surface, limit)


def _fetch_atp_winloss_index(
    index_category: str,
    time_category: str,
    limit: int,
) -> List[Dict[str, str]]:
    """
    Fetch ATP Win/Loss Index table using Playwright.
    index_category: slug from ATP_WINLOSS_INDEX_SLUGS (e.g. 'all', 'after1stsetloss', 'clay').
    time_category: 'roll' (52 Week), 'career', or 'ytd'.
    Returns list of dicts with keys: rank, player, index, titles, win_loss.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required for ATP Win/Loss Index. "
            "Install with: pip install playwright && playwright install chromium"
        ) from None

    index_category = (index_category or "all").strip().lower()
    if index_category not in ATP_WINLOSS_INDEX_SLUGS:
        index_category = "all"
    time_category = (time_category or "roll").strip().lower()
    if time_category not in ATP_WINLOSS_TIME_SLUGS:
        time_category = "roll"
    url = ATP_WINLOSS_INDEX_URL.format(
        index_category=index_category,
        time_category=time_category,
    )

    rows: List[Dict[str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=20000)
            page.wait_for_selector("table tbody tr", state="visible", timeout=15000)
            page.wait_for_timeout(1500)
            html = page.content()
        finally:
            browser.close()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return rows

    tbody = table.find("tbody")
    if not tbody:
        return rows

    for rank_one, tr in enumerate(tbody.find_all("tr"), start=1):
        tds = tr.find_all("td")
        if not tds:
            continue
        texts = [td.get_text(strip=True) for td in tds]
        # Table: Player, Index, Titles, Win/Loss (or Rank, Player, Index, Titles, Win/Loss)
        if len(texts) < 3:
            continue
        # Skip "No data available" or header-like row
        if any(nodata in " ".join(texts).lower() for nodata in ("no data", "available")):
            continue
        def _clean_player(cell: str) -> str:
            # First cell can be "1Carlos Alcaraz" or "1 Carlos Alcaraz" (rank + name in one cell)
            s = (cell or "").strip()
            s = re.sub(r"^\d+\s*", "", s)
            return s.strip()

        if len(texts) >= 5:
            row = {
                "rank": texts[0],
                "player": _clean_player(texts[1]),
                "index": texts[2],
                "titles": texts[3],
                "win_loss": texts[4],
            }
        elif len(texts) == 4:
            row = {
                "rank": str(rank_one),
                "player": _clean_player(texts[0]),
                "index": texts[1],
                "titles": texts[2],
                "win_loss": texts[3],
            }
        else:
            continue
        if (row.get("player") or "").strip():
            rows.append(row)
        if 0 < limit <= len(rows):
            break

    return rows


def _fetch_atp_tdi_leaderboard(
    time_frame: str,
    surface: str,
    limit: int,
) -> List[Dict[str, str]]:
    """
    Fetch ATP TDI Insights Leaderboard using Playwright.
    time_frame: 'average52Week' (Last 52 Weeks) or '2025'.
    surface: one of 'all', 'Clay', 'Grass', 'Hard'.
    Returns list of dicts with keys: rank, player, performance_rating, conversion_pct,
    steal_pct, serve, return_shot, forehand, backhand.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required for ATP TDI leaderboard. "
            "Install with: pip install playwright && playwright install chromium"
        ) from None

    time_frame = (time_frame or "average52Week").strip()
    if time_frame not in ATP_TDI_TIME_SLUGS:
        time_frame = "average52Week"
    surface = surface.strip() if surface else "all"
    if surface not in ATP_SERVE_SURFACES:
        surface = "all"
    url = ATP_TDI_LEADERBOARD_URL.format(time_frame=time_frame, surface=surface)

    rows: List[Dict[str, str]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=20000)
            page.wait_for_selector("table tbody tr", state="visible", timeout=15000)
            page.wait_for_timeout(1500)
            html = page.content()
        finally:
            browser.close()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return rows

    tbody = table.find("tbody")
    if not tbody:
        return rows

    def _clean_player(cell: str) -> str:
        s = (cell or "").strip()
        s = re.sub(r"^\d+\s*", "", s)
        return s.strip()

    for rank_one, tr in enumerate(tbody.find_all("tr"), start=1):
        tds = tr.find_all("td")
        if not tds:
            continue
        texts = [td.get_text(strip=True) for td in tds]
        if len(texts) < 5:
            continue
        if any(x in " ".join(texts).lower() for x in ("no results", "try a different")):
            continue
        # 9 columns: Rank, Player, Perform. Rating, Convs., Steal, Serve, Return, Forehd, Backhd
        if len(texts) >= 9:
            row = {
                "rank": texts[0],
                "player": _clean_player(texts[1]),
                "performance_rating": texts[2],
                "conversion_pct": texts[3],
                "steal_pct": texts[4],
                "serve": texts[5],
                "return_shot": texts[6],
                "forehand": texts[7],
                "backhand": texts[8],
            }
        elif len(texts) >= 8:
            row = {
                "rank": str(rank_one),
                "player": _clean_player(texts[0]),
                "performance_rating": texts[1],
                "conversion_pct": texts[2],
                "steal_pct": texts[3],
                "serve": texts[4],
                "return_shot": texts[5],
                "forehand": texts[6],
                "backhand": texts[7],
            }
        else:
            continue
        if (row.get("player") or "").strip():
            rows.append(row)
        if 0 < limit <= len(rows):
            break

    return rows


def _serve_leaderboard_worker(surface: str, limit: int) -> List[Dict[str, str]]:
    """Top-level worker for ProcessPoolExecutor; avoids asyncio/subprocess issues in Streamlit on Windows."""
    return _fetch_atp_serve_leaderboard(surface=surface, limit=limit)


def _return_leaderboard_worker(surface: str, limit: int) -> List[Dict[str, str]]:
    """Top-level worker for ProcessPoolExecutor (Return leaderboard)."""
    return _fetch_atp_return_leaderboard(surface=surface, limit=limit)


def _pressure_leaderboard_worker(surface: str, limit: int) -> List[Dict[str, str]]:
    """Top-level worker for ProcessPoolExecutor (Under Pressure leaderboard)."""
    return _fetch_atp_pressure_leaderboard(surface=surface, limit=limit)


def _winloss_index_worker(index_category: str, time_category: str, limit: int) -> List[Dict[str, str]]:
    """Top-level worker for ProcessPoolExecutor (Win/Loss Index)."""
    return _fetch_atp_winloss_index(
        index_category=index_category,
        time_category=time_category,
        limit=limit,
    )


def get_atp_winloss_index_display(
    index_category: str = "all",
    time_category: str = "roll",
    limit: int = 25,
) -> List[Dict[str, str]]:
    """Fetch ATP Win/Loss Index; when in Streamlit, runs in subprocess."""
    if "streamlit" in sys.modules:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_winloss_index_worker, index_category, time_category, limit)
            return future.result(timeout=90)
    return _fetch_atp_winloss_index(
        index_category=index_category,
        time_category=time_category,
        limit=limit,
    )


def _tdi_leaderboard_worker(time_frame: str, surface: str, limit: int) -> List[Dict[str, str]]:
    """Top-level worker for ProcessPoolExecutor (TDI leaderboard)."""
    return _fetch_atp_tdi_leaderboard(time_frame=time_frame, surface=surface, limit=limit)


def get_atp_tdi_leaderboard_display(
    time_frame: str = "average52Week",
    surface: str = "all",
    limit: int = 25,
) -> List[Dict[str, str]]:
    """Fetch ATP TDI Insights Leaderboard; when in Streamlit, runs in subprocess."""
    if "streamlit" in sys.modules:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_tdi_leaderboard_worker, time_frame, surface, limit)
            return future.result(timeout=90)
    return _fetch_atp_tdi_leaderboard(time_frame=time_frame, surface=surface, limit=limit)


def get_atp_serve_leaderboard_display(
    surface: str = "all",
    limit: int = 25,
) -> List[Dict[str, str]]:
    """
    Fetch ATP Serve leaderboard and return rows ready for display.
    surface: one of 'all', 'Clay', 'Grass', 'Hard'.
    When run inside Streamlit (e.g. on Windows), the fetch runs in a subprocess to avoid asyncio NotImplementedError.
    """
    if "streamlit" in sys.modules:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_serve_leaderboard_worker, surface, limit)
            return future.result(timeout=90)
    return _fetch_atp_serve_leaderboard(surface=surface, limit=limit)


def get_atp_return_leaderboard_display(
    surface: str = "all",
    limit: int = 25,
) -> List[Dict[str, str]]:
    """Fetch ATP Return leaderboard; when in Streamlit, runs in subprocess."""
    if "streamlit" in sys.modules:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_return_leaderboard_worker, surface, limit)
            return future.result(timeout=90)
    return _fetch_atp_return_leaderboard(surface=surface, limit=limit)


def get_atp_pressure_leaderboard_display(
    surface: str = "all",
    limit: int = 25,
) -> List[Dict[str, str]]:
    """Fetch ATP Under Pressure leaderboard; when in Streamlit, runs in subprocess."""
    if "streamlit" in sys.modules:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_pressure_leaderboard_worker, surface, limit)
            return future.result(timeout=90)
    return _fetch_atp_pressure_leaderboard(surface=surface, limit=limit)


def _print_atp_serve_leaderboard_table(rows: List[Dict[str, str]], surface: str = "all") -> None:
    if not rows:
        _print_safe("No ATP Serve leaderboard rows (page may be JS-rendered; ensure Playwright is installed).")
        return
    headers = [
        "Rank", "Player", "Serve Rating", "% 1st Serve", "% 1st Serve Pts Won",
        "% 2nd Serve Pts Won", "% Service Games Won", "Avg Aces/Match", "Avg Dbl Faults/Match",
    ]
    keys = [
        "rank", "player", "serve_rating", "pct_1st_serve", "pct_1st_serve_pts_won",
        "pct_2nd_serve_pts_won", "pct_service_games_won", "avg_aces_per_match",
        "avg_double_faults_per_match",
    ]

    def _col_width(key: str, header: str) -> int:
        return max(max(len(r.get(key) or "") for r in rows), len(header))

    widths = {h: _col_width(k, h) for h, k in zip(headers, keys)}
    sep = "-" * sum(widths.values())  # approximate

    _print_safe(f"ATP Serve Leaderboard (52 weeks, surface={surface})")
    _print_safe(sep)
    _print_safe("  ".join(h.ljust(widths[h]) for h in headers))
    _print_safe(sep)
    for r in rows:
        _print_safe("  ".join((r.get(k) or "").ljust(widths[h]) for h, k in zip(headers, keys)))


def _print_atp_return_leaderboard_table(rows: List[Dict[str, str]], surface: str = "all") -> None:
    if not rows:
        _print_safe("No ATP Return leaderboard rows.")
        return
    headers = [
        "Rank", "Player", "Return Rating", "% 1st Serve Return Pts Won",
        "% 2nd Serve Return Pts Won", "% Return Games Won", "% Break Pts Converted",
    ]
    keys = [
        "rank", "player", "return_rating", "pct_1st_serve_return_pts_won",
        "pct_2nd_serve_return_pts_won", "pct_return_games_won", "pct_break_pts_converted",
    ]
    def _col_width(key: str, header: str) -> int:
        return max(max(len(r.get(key) or "") for r in rows), len(header))
    widths = {h: _col_width(k, h) for h, k in zip(headers, keys)}
    sep = "-" * sum(widths.values())
    _print_safe(f"ATP Return Leaderboard (52 weeks, surface={surface})")
    _print_safe(sep)
    _print_safe("  ".join(h.ljust(widths[h]) for h in headers))
    _print_safe(sep)
    for r in rows:
        _print_safe("  ".join((r.get(k) or "").ljust(widths[h]) for h, k in zip(headers, keys)))


def _print_atp_pressure_leaderboard_table(rows: List[Dict[str, str]], surface: str = "all") -> None:
    if not rows:
        _print_safe("No ATP Under Pressure leaderboard rows.")
        return
    headers = [
        "Rank", "Player", "Under Pressure Rating", "% Break Pts Converted",
        "% Break Pts Saved", "% Tie Breaks Won", "% Deciding Sets Won",
    ]
    keys = [
        "rank", "player", "pressure_rating", "pct_break_pts_converted",
        "pct_break_pts_saved", "pct_tiebreaks_won", "pct_deciding_sets_won",
    ]
    def _col_width(key: str, header: str) -> int:
        return max(max(len(r.get(key) or "") for r in rows), len(header))
    widths = {h: _col_width(k, h) for h, k in zip(headers, keys)}
    sep = "-" * sum(widths.values())
    _print_safe(f"ATP Under Pressure Leaderboard (52 weeks, surface={surface})")
    _print_safe(sep)
    _print_safe("  ".join(h.ljust(widths[h]) for h in headers))
    _print_safe(sep)
    for r in rows:
        _print_safe("  ".join((r.get(k) or "").ljust(widths[h]) for h, k in zip(headers, keys)))


def _print_atp_winloss_index_table(
    rows: List[Dict[str, str]],
    index_category: str = "all",
    time_category: str = "roll",
) -> None:
    if not rows:
        _print_safe("No ATP Win/Loss Index rows.")
        return
    category_label = next((label for slug, label in ATP_WINLOSS_INDEX_CATEGORIES if slug == index_category), index_category)
    time_label = next((label for slug, label in ATP_WINLOSS_TIME_CATEGORIES if slug == time_category), time_category)
    headers = ["Rank", "Player", "Index", "Titles", "Win/Loss"]
    keys = ["rank", "player", "index", "titles", "win_loss"]

    def _col_width(key: str, header: str) -> int:
        return max(max(len(r.get(key) or "") for r in rows), len(header))

    widths = {h: _col_width(k, h) for h, k in zip(headers, keys)}
    sep = "-" * sum(widths.values())
    _print_safe(f"ATP Win/Loss Index — {category_label} ({time_label})")
    _print_safe(sep)
    _print_safe("  ".join(h.ljust(widths[h]) for h in headers))
    _print_safe(sep)
    for r in rows:
        _print_safe("  ".join((r.get(k) or "").ljust(widths[h]) for h, k in zip(headers, keys)))


def _print_atp_tdi_leaderboard_table(
    rows: List[Dict[str, str]],
    time_frame: str = "average52Week",
    surface: str = "all",
) -> None:
    if not rows:
        _print_safe("No ATP TDI Insights Leaderboard rows.")
        return
    time_label = next((label for slug, label in ATP_TDI_TIME_FRAMES if slug == time_frame), time_frame)
    headers = [
        "Rank", "Player", "Perf. Rating", "Convs.%", "Steal%",
        "Serve", "Return", "Forehand", "Backhand",
    ]
    keys = [
        "rank", "player", "performance_rating", "conversion_pct", "steal_pct",
        "serve", "return_shot", "forehand", "backhand",
    ]

    def _col_width(key: str, header: str) -> int:
        return max(max(len(r.get(key) or "") for r in rows), len(header))

    widths = {h: _col_width(k, h) for h, k in zip(headers, keys)}
    sep = "-" * sum(widths.values())
    _print_safe(f"ATP TDI Insights Leaderboard — {time_label}, surface={surface}")
    _print_safe(sep)
    _print_safe("  ".join(h.ljust(widths[h]) for h in headers))
    _print_safe(sep)
    for r in rows:
        _print_safe("  ".join((r.get(k) or "").ljust(widths[h]) for h, k in zip(headers, keys)))


def _build_upcoming_matchups(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Build list of matchup dicts for query mode (tennis etc.).
    Returns (matchups, message) where message is the "Found N series..." line.
    """
    query_lower = (args.query or "").strip().lower()
    # For tennis, skip the global market scan (no direct "tennis" hits there); go straight to series.
    if query_lower != "tennis":
        all_markets = list_markets(status=args.status, limit=args.limit, max_pages=args.max_pages)
        hits = [m for m in all_markets if isinstance(m, dict) and matches_query(m, args.query)]
        if hits:
            return [], "direct"

    series_pages = getattr(args, "series_pages", 10)
    if query_lower == "tennis":
        series_pages = max(series_pages, 8)  # At least 8 pages; UI can pass 10 for speed
    all_series = list_series(limit=args.limit, max_pages=series_pages)
    if query_lower == "tennis":
        # Include all tennis circuits: WTA, ATP, Challenger, 125K, Indian Wells, etc.
        seen_tickers: set = set()
        series_hits = []
        for s in all_series:
            if not isinstance(s, dict):
                continue
            st = s.get("ticker")
            if not isinstance(st, str) or st in seen_tickers:
                continue
            if series_matches_tennis(s):
                seen_tickers.add(st)
                series_hits.append(s)
    else:
        series_hits = [s for s in all_series if isinstance(s, dict) and series_matches_query(s, args.query)]

    num_markets_scanned = len(all_markets) if query_lower != "tennis" else 0
    msg = (
        f"Found {len(series_hits)} matching series; fetching markets..."
        if query_lower == "tennis"
        else f"No direct matches in first {num_markets_scanned} markets. Found {len(series_hits)} series; fetching..."
    )
    if args.output == "text":
        _print_safe(msg)

    market_pages_per_series = getattr(args, "market_pages_per_series", None)
    if market_pages_per_series is None:
        market_pages_per_series = 2 if query_lower == "tennis" else 1
    markets_by_event: Dict[str, List[Dict[str, Any]]] = {}

    def fetch_series_markets(s: Dict[str, Any]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        st = s.get("ticker") if isinstance(s, dict) else None
        if not isinstance(st, str) or not st:
            return None, []
        ms = get_markets(series_ticker=st, status=args.status, limit=args.limit, max_pages=market_pages_per_series)
        return st, ms

    if query_lower == "tennis" and len(series_hits) > 1 and getattr(args, "parallel_series", True):
        with ThreadPoolExecutor(max_workers=min(MAX_SERIES_WORKERS, len(series_hits))) as ex:
            futures = {ex.submit(fetch_series_markets, s): s for s in series_hits}
            for fut in as_completed(futures):
                try:
                    _st, ms = fut.result()
                    for m in ms:
                        et = m.get("event_ticker")
                        if isinstance(et, str) and et:
                            markets_by_event.setdefault(et, []).append(m)
                except Exception:
                    pass
    else:
        for i, s in enumerate(series_hits):
            if args.output == "text":
                _print_safe(f"Fetching series {i + 1}/{len(series_hits)}...")
            _st, ms = fetch_series_markets(s)
            for m in ms:
                et = m.get("event_ticker")
                if isinstance(et, str) and et:
                    markets_by_event.setdefault(et, []).append(m)
            time.sleep(SERIES_DELAY_S)

    now = datetime.now(timezone.utc)
    local_now = now.astimezone()
    horizon = now + timedelta(days=max(args.days, 0))
    local_horizon = horizon.astimezone()

    def event_sort_key(item: Tuple[str, List[Dict[str, Any]]]) -> datetime:
        _, markets = item
        times = [
            parse_iso8601_utc(mm.get("expected_expiration_time"))
            for mm in markets
            if isinstance(mm, dict)
        ]
        times = [t for t in times if t is not None]
        return min(times) if times else datetime.max.replace(tzinfo=timezone.utc)

    event_items = sorted(markets_by_event.items(), key=event_sort_key)
    matchups: List[Dict[str, Any]] = []

    for event_ticker, ev_markets in event_items:
        times = [
            parse_iso8601_utc(m.get("expected_expiration_time"))
            for m in ev_markets
            if isinstance(m, dict)
        ]
        times = [t for t in times if t is not None]
        expected = min(times) if times else None
        if expected is not None:
            expected_local = expected.astimezone()
            if expected > horizon:
                continue
            # Keep already-started matches that are still on today's open market tape.
            if expected_local.date() < local_now.date():
                continue
            if expected_local.date() > local_horizon.date():
                continue

        # Only match-winner markets: skip prop markets (Set, Game Spread, Exact Score) by market title
        prop_markers = (": Set", ": Game Spread", ": Exact Match Score")
        player_prices: Dict[str, float] = {}
        raw_contract_tickers: List[str] = []
        for m in ev_markets:
            title = (m.get("title") or "") if isinstance(m, dict) else ""
            if any(mark in title for mark in prop_markers):
                continue
            player = player_from_market_title(m)
            yes_bid = to_float(m.get("yes_bid_dollars"))
            if isinstance(m, dict) and m.get("ticker"):
                raw_contract_tickers.append(str(m.get("ticker")))
            if player and yes_bid is not None:
                # If multiple contracts map to same player within an event, keep the latest price.
                player_prices[player] = yes_bid

        if len(player_prices) < 2:
            continue
        if len(player_prices) > 2:
            continue

        # Derive title from player names to avoid one get_event API call per matchup
        ev_title = " vs ".join(sorted(player_prices.keys(), key=lambda x: x.lower()))
        event_meta: Dict[str, Any] = {}
        start_state = infer_market_start_state(expected.isoformat().replace("+00:00", "Z") if expected else None, now=now)
        sample_market = ev_markets[0] if ev_markets else {}
        inferred_round = extract_tennis_round(
            sample_market.get("title") if isinstance(sample_market, dict) else None,
            sample_market.get("rules_primary") if isinstance(sample_market, dict) else None,
            sample_market.get("rules_secondary") if isinstance(sample_market, dict) else None,
        )
        if getattr(args, "fetch_event_title", False):
            try:
                ev = get_event(event_ticker).get("event", {})
                if isinstance(ev, dict):
                    ev_title = safe_text(ev.get("title") or ev.get("sub_title") or ev_title)
            except Exception:
                pass
        if getattr(args, "fetch_event_metadata", True):
            try:
                event_payload = get_event(event_ticker)
                event_meta = summarize_tennis_event(
                    event_payload,
                    fallback_expected_expiration_time=expected.isoformat().replace("+00:00", "Z") if expected else None,
                    now=now,
                )
                ev_title = event_meta.get("event_title") or ev_title
            except Exception:
                event_meta = {}

        if "/" in ev_title:
            continue

        players_sorted = sorted(player_prices.items(), key=lambda kv: kv[0].lower())
        p_sum = sum(p for _, p in players_sorted) or 1.0
        when = expected.isoformat().replace("+00:00", "Z") if expected else "unknown time"
        when_display = format_iso8601_local(when) or when

        players_out = [
            {
                "name": name,
                "yes_bid": price,
                "pct": f"{price:.1%}",
                "normalized_pct": f"{(price / p_sum):.1%}",
            }
            for name, price in players_sorted
        ]
        # Canonical key: event ticker + sorted player names + event date (not exact time)
        canonical_key = (
            event_ticker,
            tuple(n.lower() for n, _ in players_sorted),
            (when or "").split("T")[0],
        )
        # Basic validation flags
        names_only = [n for n, _ in players_sorted]
        market_valid = len(names_only) == 2 and names_only[0].strip().lower() != names_only[1].strip().lower()
        low_liquidity = all(price < 0.05 for _, price in players_sorted)

        matchups.append({
            "event": ev_title,
            "event_ticker": event_ticker,
            "when": when,
            "when_display": when_display,
            "scheduled_start": event_meta.get("scheduled_start") or when,
            "scheduled_start_display": event_meta.get("scheduled_start_local") or when_display,
            "is_started": event_meta.get("is_started", start_state.get("is_started")),
            "market_phase": event_meta.get("market_phase") or start_state.get("phase"),
            "market_phase_label": event_meta.get("market_phase_label") or start_state.get("phase_label"),
            "market_phase_detail": event_meta.get("market_phase_detail") or start_state.get("phase_detail"),
            "market_phase_source": event_meta.get("market_phase_source") or start_state.get("inference_source"),
            "event_subtitle": event_meta.get("event_subtitle", ""),
            "event_category": event_meta.get("event_category", ""),
            "event_series_ticker": event_meta.get("event_series_ticker", ""),
            "event_competition": event_meta.get("event_competition", ""),
            "event_scope": event_meta.get("event_scope", ""),
            "event_round": event_meta.get("event_round") or inferred_round,
            "event_info_line": event_meta.get("event_info_line") or " · ".join(
                part for part in [event_meta.get("event_competition", ""), inferred_round] if part
            ),
            "players": players_out,
            "canonical_key": canonical_key,
            "raw_contract_tickers": raw_contract_tickers,
            "market_valid": market_valid,
            "low_liquidity": low_liquidity,
        })

    matchups = dedupe_matchup_rows(matchups)
    return matchups[: max(args.max_results, 0)], msg


def get_tennis_matchups(
    player: Optional[str] = None,
    days: int = 3,
    max_results: int = 50,
    *,
    fast: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetch upcoming tennis match-winner matchups. If player is given (e.g. "zverev"),
    only matchups involving that player; use "all" or None for every match.
    When fast=True (default for UI): fewer series/market pages, parallel fetch, no event-title API calls.
    Returns (matchups, message) for use by CLI or UI.
    """
    ns = argparse.Namespace(
        query="tennis",
        player=player or "",
        status="open",
        limit=100,
        max_pages=3,
        series_pages=10 if fast else 20,
        market_pages_per_series=2 if fast else 5,
        parallel_series=fast,
        fetch_event_title=not fast,
        fetch_event_metadata=not fast,
        max_results=max_results,
        days=days,
        output="text",
    )
    matchups, msg = _build_upcoming_matchups(ns)
    player_arg = (player or "").strip()
    if player_arg and player_arg.lower() != "all":
        needle = player_arg.lower()
        matchups = [m for m in matchups if any(needle in p["name"].lower() for p in m["players"])]
    return matchups, msg


def dedupe_matchup_rows(matchups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove repeated matchup rows while keeping the richer row when duplicates collide."""
    best_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    order: List[Tuple[Any, ...]] = []

    def _fallback_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        players = row.get("players") or []
        names = tuple(sorted(str(p.get("name") or "").strip().lower() for p in players))
        when = str(row.get("scheduled_start") or row.get("when") or "")
        return (
            str(row.get("event_ticker") or row.get("event") or "").strip().lower(),
            names,
            when.split("T")[0],
        )

    def _score(row: Dict[str, Any]) -> Tuple[int, int, int]:
        info_line = str(row.get("event_info_line") or "").strip()
        round_name = str(row.get("event_round") or "").strip()
        phase = str(row.get("market_phase") or "").strip()
        tickers = row.get("raw_contract_tickers") or []
        return (
            len(tickers),
            int(bool(info_line and info_line.lower() != "unknown")),
            int(bool(round_name or phase)),
        )

    for row in matchups:
        key = row.get("canonical_key") or _fallback_key(row)
        if key not in best_by_key:
            best_by_key[key] = row
            order.append(key)
            continue
        if _score(row) > _score(best_by_key[key]):
            best_by_key[key] = row
    return [best_by_key[key] for key in order]


def enrich_matchups_with_predictions(
    matchups: List[Dict[str, Any]],
    atp_datasets: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    model_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Enrich each matchup's players with model_prob, kalshi_prob, edge.
    atp_datasets can have keys: atp_rankings, serve_leaderboard, return_leaderboard,
    pressure_leaderboard, winloss_index, tdi_leaderboard (each a list of row dicts).
    Kalshi implied prob is taken from yes_bid (price in [0,1] per share).
    Returns (matchups, used_fallback). Mutates matchups in place.
    """
    used_fallback = True
    if not _prediction_available:
        return matchups, used_fallback

    datasets = atp_datasets or {}
    player_db = datasets.get("player_db")  # saved DB (list of player records)
    atp_rankings = datasets.get("atp_rankings")
    serve_rows = datasets.get("serve_leaderboard")
    return_rows = datasets.get("return_leaderboard")
    pressure_rows = datasets.get("pressure_leaderboard")
    winloss_rows = datasets.get("winloss_index")
    tdi_rows = datasets.get("tdi_leaderboard")

    model_core: Any = None
    model_enhanced: Any = None
    model_legacy: Any = None
    feature_names_core: Optional[List[str]] = None
    feature_names_enhanced: Optional[List[str]] = None
    feature_names_legacy: Optional[List[str]] = None
    medians_core: Dict[str, Any] = {}
    medians_enhanced: Dict[str, Any] = {}
    medians_legacy: Dict[str, Any] = {}

    if model_path:
        try:
            model_legacy, feature_names_legacy, medians_legacy = load_model(model_path)
            used_fallback = False
        except (FileNotFoundError, OSError):
            pass

    try:
        _cp = get_default_core_path()
        if _cp and Path(_cp).exists():
            model_core, feature_names_core, medians_core = load_model(_cp)
            used_fallback = False
    except (FileNotFoundError, OSError):
        pass

    try:
        _ep = get_default_enhanced_path()
        if _ep and Path(_ep).exists():
            model_enhanced, feature_names_enhanced, medians_enhanced = load_model(_ep)
            used_fallback = False
    except (FileNotFoundError, OSError):
        pass

    if model_legacy is None:
        try:
            default_path = get_default_model_path()
            if default_path and Path(default_path).exists():
                model_legacy, feature_names_legacy, medians_legacy = load_model(default_path)
                used_fallback = False
        except (FileNotFoundError, OSError):
            pass

    medians_union: Dict[str, Any] = {}
    for d in (medians_core, medians_enhanced, medians_legacy):
        if d:
            medians_union.update(d)

    def _kalshi_prob(yes_bid: Any) -> float:
        v = to_float(yes_bid)
        if v is None or v < 0:
            return 0.0
        return max(0.0, min(1.0, float(v)))

    def _two_sided_market(yes_a: Any, yes_b: Any) -> Any:
        if TwoSidedQuote is not None:
            return TwoSidedQuote.from_yes_bids(yes_a, yes_b)
        raw_a = _kalshi_prob(yes_a)
        raw_b = _kalshi_prob(yes_b)
        tot = raw_a + raw_b
        if tot > 0:
            return type("Q", (), {"implied_a": raw_a / tot, "implied_b": raw_b / tot, "fair_a": raw_a / tot, "fair_b": raw_b / tot, "quality": None})()
        return type("Q", (), {"implied_a": 0.5, "implied_b": 0.5, "fair_a": 0.5, "fair_b": 0.5, "quality": None})()

    # Optional tournament catalog for richer context (shared with PvP)
    try:
        from deuce.tournament_catalog import DEFAULT_CATALOG_PATH, load_catalog

        _catalog_path = Path(DEFAULT_CATALOG_PATH)
        if not _catalog_path.is_absolute():
            _catalog_path = Path.cwd() / _catalog_path
        _catalog = load_catalog(str(_catalog_path)) if _catalog_path.exists() else None
    except Exception:
        _catalog = None

    for m in matchups:
        players = m.get("players") or []
        if len(players) != 2:
            continue
        name_a, name_b = players[0]["name"], players[1]["name"]
        stats_a = build_player_stats_from_datasets(
            name_a,
            player_db=player_db,
            atp_rankings=atp_rankings,
            serve_rows=serve_rows,
            return_rows=return_rows,
            pressure_rows=pressure_rows,
            winloss_rows=winloss_rows,
            tdi_rows=tdi_rows,
        )
        stats_b = build_player_stats_from_datasets(
            name_b,
            player_db=player_db,
            atp_rankings=atp_rankings,
            serve_rows=serve_rows,
            return_rows=return_rows,
            pressure_rows=pressure_rows,
            winloss_rows=winloss_rows,
            tdi_rows=tdi_rows,
        )
        # Best-effort tournament name for catalog lookup (reuse event title field)
        tournament_name = m.get("event")
        has_tournament_context = bool(tournament_name and _catalog)

        try:
            result = predict_match_probability(
                stats_a,
                stats_b,
                model=model_legacy,
                model_core=model_core,
                model_enhanced=model_enhanced,
                feature_names=None,
                tournament_name=tournament_name if has_tournament_context else None,
                catalog=_catalog if has_tournament_context else None,
                medians=medians_union or None,
            )
            if result.confidence == "model":
                used_fallback = False
        except Exception:
            continue
        prob_a = result.model_prob_a
        prob_b = result.model_prob_b
        quote = _two_sided_market(players[0].get("yes_bid"), players[1].get("yes_bid"))
        market_prob_a_raw = _kalshi_prob(players[0].get("yes_bid"))
        market_prob_b_raw = _kalshi_prob(players[1].get("yes_bid"))
        market_prob_a = float(getattr(quote, "implied_a", 0.5))
        market_prob_b = float(getattr(quote, "implied_b", 0.5))
        fair_a = float(getattr(quote, "fair_a", market_prob_a))
        fair_b = float(getattr(quote, "fair_b", market_prob_b))
        mkt_quality = getattr(quote, "quality", None)
        # Trust / information quality heuristics
        completeness = getattr(result, "data_completeness", None)
        prediction_source = getattr(result, "model_used", result.confidence)
        low_information = False
        if str(prediction_source) == "fallback":
            low_information = True
        if completeness is not None and completeness < 0.5:
            low_information = True
        if not has_tournament_context:
            low_information = True
        # Extremely low liquidity: both sides thin
        low_liquidity = market_prob_a_raw < 0.05 and market_prob_b_raw < 0.05
        if str(getattr(mkt_quality, "value", mkt_quality) or "") in ("thin", "malformed"):
            low_liquidity = True

        # Attach per-player probabilities and edge, plus prediction source label.
        for i, p in enumerate(players):
            kalshi = market_prob_a if i == 0 else market_prob_b
            fair_side = fair_a if i == 0 else fair_b
            model_p = prob_a if i == 0 else prob_b
            edge = model_p - fair_side
            p["model_prob"] = model_p
            p["kalshi_prob"] = kalshi
            p["market_prob"] = kalshi
            p["market_fair_prob"] = fair_side
            p["market_prob_raw"] = market_prob_a_raw if i == 0 else market_prob_b_raw
            p["edge"] = edge
            p["prediction_source"] = prediction_source
            p["low_information_prediction"] = low_information
            p["edge_reliable"] = (not low_information) and (not low_liquidity)
        # Attach market-level prediction metadata for UI/debug.
        m["prediction_source"] = prediction_source
        m["low_information_prediction"] = low_information
        m["edge_reliable"] = (not low_information) and (not low_liquidity)
        m["has_tournament_context"] = has_tournament_context
        m["low_liquidity"] = low_liquidity
        m["market_quality"] = str(getattr(mkt_quality, "value", mkt_quality) or "")
        lb_cov = float(getattr(result, "leaderboard_coverage_score", 0.0) or 0.0)
        m["deuce"] = {
            "route": result.model_used,
            "data_completeness": completeness,
            "leaderboard_coverage_score": lb_cov,
            "evidence_quality": getattr(result, "evidence_quality", None),
            "warnings": list(getattr(result, "warnings", None) or []),
            "confidence": getattr(result, "confidence", None),
        }
        _used = result.model_used
        if _used == "core" and model_core is not None:
            _ev_model, _ev_fn = model_core, feature_names_core
        elif _used == "enhanced" and model_enhanced is not None:
            _ev_model, _ev_fn = model_enhanced, feature_names_enhanced
        else:
            _ev_model, _ev_fn = model_legacy, feature_names_legacy
        if evaluate_prediction is not None:
            try:
                eval_result = evaluate_prediction(
                    result,
                    market_prob_a=fair_a,
                    market_prob_b=fair_b,
                    model=_ev_model,
                    feature_names=_ev_fn,
                )
                m["evaluation"] = eval_result
            except Exception:
                pass

    return matchups, used_fallback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch unauthenticated Kalshi public market data (series mode or search mode), or scrape ATP singles rankings."
    )
    parser.add_argument(
        "--series",
        default=None,
        help="Series ticker (e.g. KXHIGHNY). If omitted, use --query to search.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help='Search term (e.g. "tennis") to filter markets/series.',
    )
    parser.add_argument(
        "--player",
        default=None,
        metavar="NAME",
        help='Player name (case-insensitive, e.g. zverev) or "all" for every match. Implies tennis if --query not set. Odds in $.',
    )
    parser.add_argument(
        "--status",
        default="open",
        choices=["open", "all", "closed", "settled"],
        help="Market status filter.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Markets/series page size.")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pages for /markets (cursor).")
    parser.add_argument(
        "--series-pages",
        type=int,
        default=10,
        help="Max pages for /series when using --query fallback.",
    )
    parser.add_argument("--max-results", type=int, default=50, help="Max matchups to return in search mode.")
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Only include events expiring in the next N days (search mode).",
    )
    parser.add_argument(
        "--output",
        default="text",
        choices=["text", "json", "csv"],
        help="Output format: text (human), json, or csv.",
    )
    parser.add_argument("--orderbook-levels", type=int, default=5, help="Top N orderbook levels (series mode).")
    parser.add_argument("--no-event", action="store_true", help="Skip event details (series mode).")
    parser.add_argument("--no-orderbook", action="store_true", help="Skip orderbook (series mode).")
    parser.add_argument(
        "--atp-rankings",
        action="store_true",
        help="Scrape ATP Tour singles rankings and print a terminal table (Rank, Player, Age, Official Points, +/-, Tourn Played, Dropping, Next Best).",
    )
    parser.add_argument(
        "--atp-limit",
        type=int,
        default=25,
        help="Number of ATP ranking rows to display when using --atp-rankings.",
    )
    parser.add_argument(
        "--atp-serve",
        action="store_true",
        help="Fetch ATP Serve leaderboard (52 weeks) and print table.",
    )
    parser.add_argument(
        "--atp-serve-surface",
        type=str,
        default="all",
        choices=list(ATP_SERVE_SURFACES),
        help="Surface for ATP Serve leaderboard: all, Clay, Grass, Hard.",
    )
    parser.add_argument(
        "--atp-serve-limit",
        type=int,
        default=25,
        help="Number of ATP Serve leaderboard rows when using --atp-serve.",
    )
    parser.add_argument(
        "--atp-return",
        action="store_true",
        help="Fetch ATP Return leaderboard (52 weeks) and print table.",
    )
    parser.add_argument(
        "--atp-return-surface",
        type=str,
        default="all",
        choices=list(ATP_SERVE_SURFACES),
        help="Surface for ATP Return leaderboard.",
    )
    parser.add_argument(
        "--atp-return-limit",
        type=int,
        default=25,
        help="Number of ATP Return leaderboard rows when using --atp-return.",
    )
    parser.add_argument(
        "--atp-pressure",
        action="store_true",
        help="Fetch ATP Under Pressure leaderboard (52 weeks) and print table.",
    )
    parser.add_argument(
        "--atp-pressure-surface",
        type=str,
        default="all",
        choices=list(ATP_SERVE_SURFACES),
        help="Surface for ATP Under Pressure leaderboard.",
    )
    parser.add_argument(
        "--atp-pressure-limit",
        type=int,
        default=25,
        help="Number of ATP Under Pressure leaderboard rows when using --atp-pressure.",
    )
    parser.add_argument(
        "--atp-winloss",
        action="store_true",
        help="Fetch ATP Win/Loss Index and print table.",
    )
    parser.add_argument(
        "--atp-winloss-category",
        type=str,
        default="all",
        choices=ATP_WINLOSS_INDEX_SLUGS,
        help="Win/Loss Index category (e.g. all, after1stsetloss, clay, tiebreak).",
    )
    parser.add_argument(
        "--atp-winloss-time",
        type=str,
        default="roll",
        choices=ATP_WINLOSS_TIME_SLUGS,
        help="Time period: roll (52 Week), career, ytd.",
    )
    parser.add_argument(
        "--atp-winloss-limit",
        type=int,
        default=25,
        help="Number of Win/Loss Index rows when using --atp-winloss.",
    )
    parser.add_argument(
        "--atp-tdi",
        action="store_true",
        help="Fetch ATP TDI Insights Leaderboard and print table.",
    )
    parser.add_argument(
        "--atp-tdi-time",
        type=str,
        default="average52Week",
        choices=ATP_TDI_TIME_SLUGS,
        help="TDI time frame: average52Week (Last 52 Weeks), 2025.",
    )
    parser.add_argument(
        "--atp-tdi-surface",
        type=str,
        default="all",
        choices=list(ATP_SERVE_SURFACES),
        help="Surface for TDI leaderboard.",
    )
    parser.add_argument(
        "--atp-tdi-limit",
        type=int,
        default=25,
        help="Number of TDI leaderboard rows when using --atp-tdi.",
    )
    parser.add_argument(
        "--update-player-db",
        action="store_true",
        help="Fetch ATP rankings and all leaderboards, merge into player DB, then exit.",
    )
    parser.add_argument(
        "--player-db-path",
        type=str,
        default=PLAYER_DB_DEFAULT_PATH if _player_db_available else "data/player_stats.json",
        help="Path to player stats JSON (used with --update-player-db).",
    )
    parser.add_argument(
        "--fix-player-db",
        action="store_true",
        help="Strip leading rank movement from player names in DB, dedupe, save, then exit.",
    )
    args = parser.parse_args()

    # --player: filter to that player's match(es); default to tennis if no --query
    if args.player and not args.query:
        args.query = "tennis"

    if getattr(args, "fix_player_db", False) and _player_db_available:
        from deuce.player_db import fix_player_names_and_save
        n = fix_player_names_and_save(args.player_db_path)
        _print_safe(f"Player DB fixed and saved: {n} players.")
        return 0

    if getattr(args, "update_player_db", False) and _player_db_available:
        # Fetch all ATP data and merge into player DB
        _print_safe("Fetching ATP rankings and leaderboards for player DB update...")
        try:
            atp_rankings = _fetch_atp_rankings(limit=200)
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP rankings: {exc}")
            atp_rankings = None
        try:
            serve_rows = _fetch_atp_serve_leaderboard(surface="all", limit=100)
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching Serve leaderboard: {exc}")
            serve_rows = None
        try:
            return_rows = _fetch_atp_return_leaderboard(surface="all", limit=100)
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching Return leaderboard: {exc}")
            return_rows = None
        try:
            pressure_rows = _fetch_atp_pressure_leaderboard(surface="all", limit=100)
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching Pressure leaderboard: {exc}")
            pressure_rows = None
        try:
            winloss_rows = _fetch_atp_winloss_index(
                index_category="all", time_category="roll", limit=100
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching Win/Loss Index: {exc}")
            winloss_rows = None
        try:
            tdi_rows = _fetch_atp_tdi_leaderboard(
                time_frame="average52Week", surface="all", limit=100
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching TDI leaderboard: {exc}")
            tdi_rows = None
        if atp_rankings:
            _normalize_atp_rows(atp_rankings)
        n = update_db_from_atp_datasets(
            args.player_db_path,
            atp_rankings=atp_rankings,
            serve_leaderboard=serve_rows,
            return_leaderboard=return_rows,
            pressure_leaderboard=pressure_rows,
            winloss_index=winloss_rows,
            tdi_leaderboard=tdi_rows,
        )
        _print_safe(f"Player database updated: {n} players saved.")
        return 0

    if args.atp_rankings:
        try:
            rows = _fetch_atp_rankings(limit=max(args.atp_limit, 1))
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP rankings from {ATP_SINGLES_RANKINGS_URL}: {exc}")
            return 1
        _print_atp_rankings_table(rows)
        return 0

    if args.atp_serve:
        try:
            rows = _fetch_atp_serve_leaderboard(
                surface=args.atp_serve_surface,
                limit=max(args.atp_serve_limit, 1),
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP Serve leaderboard: {exc}")
            return 1
        _print_atp_serve_leaderboard_table(rows, surface=args.atp_serve_surface)
        return 0

    if args.atp_return:
        try:
            rows = _fetch_atp_return_leaderboard(
                surface=args.atp_return_surface,
                limit=max(args.atp_return_limit, 1),
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP Return leaderboard: {exc}")
            return 1
        _print_atp_return_leaderboard_table(rows, surface=args.atp_return_surface)
        return 0

    if args.atp_pressure:
        try:
            rows = _fetch_atp_pressure_leaderboard(
                surface=args.atp_pressure_surface,
                limit=max(args.atp_pressure_limit, 1),
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP Under Pressure leaderboard: {exc}")
            return 1
        _print_atp_pressure_leaderboard_table(rows, surface=args.atp_pressure_surface)
        return 0

    if args.atp_winloss:
        try:
            rows = _fetch_atp_winloss_index(
                index_category=args.atp_winloss_category,
                time_category=args.atp_winloss_time,
                limit=max(args.atp_winloss_limit, 1),
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP Win/Loss Index: {exc}")
            return 1
        _print_atp_winloss_index_table(
            rows,
            index_category=args.atp_winloss_category,
            time_category=args.atp_winloss_time,
        )
        return 0

    if args.atp_tdi:
        try:
            rows = _fetch_atp_tdi_leaderboard(
                time_frame=args.atp_tdi_time,
                surface=args.atp_tdi_surface,
                limit=max(args.atp_tdi_limit, 1),
            )
        except Exception as exc:  # noqa: BLE001
            _print_safe(f"Error fetching ATP TDI leaderboard: {exc}")
            return 1
        _print_atp_tdi_leaderboard_table(
            rows,
            time_frame=args.atp_tdi_time,
            surface=args.atp_tdi_surface,
        )
        return 0

    if args.query:
        # Direct market hits: print raw list (text only for now).
        all_markets = list_markets(status=args.status, limit=args.limit, max_pages=args.max_pages)
        hits = [m for m in all_markets if isinstance(m, dict) and matches_query(m, args.query)]
        if hits:
            if args.output != "text":
                # JSON/CSV for direct hits: one row per market
                out = [
                    {
                        "ticker": m.get("ticker"),
                        "title": m.get("title"),
                        "yes_bid_dollars": m.get("yes_bid_dollars"),
                        "no_bid_dollars": m.get("no_bid_dollars"),
                        "volume_fp": m.get("volume_fp"),
                    }
                    for m in hits[: max(args.max_results, 0)]
                ]
                if args.output == "json":
                    print(json.dumps({"markets": out}, indent=2))
                else:
                    w = csv.DictWriter(sys.stdout, fieldnames=["ticker", "title", "yes_bid_dollars", "no_bid_dollars", "volume_fp"])
                    w.writeheader()
                    w.writerows(out)
            else:
                _print_safe(f'Markets matching "{args.query}" (status={args.status}): {len(hits)}')
                for m in hits[: max(args.max_results, 0)]:
                    _print_safe(f"- {safe_text(m.get('ticker'))}: {safe_text(m.get('title'))}")
                    _print_safe(f"  Yes Bid: ${safe_text(m.get('yes_bid_dollars'))} | No Bid: ${safe_text(m.get('no_bid_dollars'))} | Volume: {safe_text(m.get('volume_fp'))}")
            return 0

        matchups, _ = _build_upcoming_matchups(args)
        if not matchups:
            if args.output == "text":
                _print_safe(f"No upcoming events found in the next {args.days} days for that query.")
            elif args.output == "json":
                print(json.dumps({"matchups": [], "count": 0}))
            elif args.output == "csv":
                w = csv.writer(sys.stdout)
                w.writerow(["event", "when", "player", "yes_bid", "pct", "normalized_pct"])
            return 0

        # Filter by player name (case-insensitive) unless --player all
        player_arg = (args.player or "").strip()
        if player_arg and player_arg.lower() != "all":
            needle = player_arg.lower()
            matchups = [
                m for m in matchups
                if any(needle in p["name"].lower() for p in m["players"])
            ]
            if args.output == "text" and matchups:
                _print_safe(f"Match(es) involving \"{player_arg}\" (odds in $ per share):\n")

        if not matchups:
            if args.output == "text":
                _print_safe(f"No matchups found for player \"{player_arg}\" in the next {args.days} days." if player_arg and player_arg.lower() != "all" else f"No upcoming match-winner events in the next {args.days} days.")
            elif args.output == "json":
                print(json.dumps({"matchups": [], "count": 0}))
            elif args.output == "csv":
                w = csv.writer(sys.stdout)
                w.writerow(["event", "when", "player", "yes_bid", "pct", "normalized_pct"])
            return 0

        if args.output == "json":
            print(json.dumps({"matchups": matchups, "count": len(matchups)}, indent=2))
            return 0
        if args.output == "csv":
            w = csv.writer(sys.stdout)
            w.writerow(["event", "when", "player", "yes_bid", "pct", "normalized_pct"])
            for m in matchups:
                for p in m["players"]:
                    w.writerow([m["event"], m["when"], p["name"], p["yes_bid"], p["pct"], p["normalized_pct"]])
            return 0

        for m in matchups:
            _print_safe(f"- {m['event']} ({m['when']})")
            for p in m["players"]:
                # Odds in $: cost per share to buy YES (this player wins)
                line = f"  {p['name']}: buy YES at ${p['yes_bid']:.2f} per share  (implied win {p['pct']})"
                if "model_prob" in p and "edge" in p:
                    line += f"  model={p['model_prob']:.1%}  edge={p['edge']:+.1%}"
                _print_safe(line)

        # Footer
        _print_safe(f"\n{len(matchups)} match(es) in the next {args.days} days.")
        _print_safe(f"Browse tennis markets: {MARKETS_UI_BASE}/kxwtamatch")
        return 0

    # --- Series mode ---
    series_ticker = args.series or "KXHIGHNY"
    series_data = get_series(series_ticker)
    series = series_data.get("series", {})
    print(f"Series Title: {safe_text(series.get('title'))}")
    print(f"Frequency: {safe_text(series.get('frequency'))}")
    print(f"Category: {safe_text(series.get('category'))}")

    markets = get_markets(series_ticker=series_ticker, status=args.status, limit=args.limit, max_pages=args.max_pages)
    if not markets and args.status == "open":
        print("\nNo open markets found. Retrying with status=all...")
        markets = get_markets(series_ticker=series_ticker, status="all", limit=args.limit, max_pages=args.max_pages)

    print(f"\nMarkets in {series_ticker} (status={args.status}): {len(markets)}")
    for m in markets[:10]:
        print(f"- {safe_text(m.get('ticker'))}: {safe_text(m.get('title'))}")
        print(f"  Event: {safe_text(m.get('event_ticker'))}")
        print(f"  Yes Price: ${safe_text(m.get('yes_bid_dollars'))} | Volume: {safe_text(m.get('volume_fp'))}")
        print()

    if not markets:
        raise SystemExit("No markets returned. Try a different --series.")

    first = markets[0]
    market_ticker = first.get("ticker")
    event_ticker = first.get("event_ticker")
    if not isinstance(market_ticker, str) or not market_ticker:
        raise SystemExit("Unexpected market payload: missing ticker.")

    if not args.no_event and isinstance(event_ticker, str) and event_ticker:
        event_data = get_event(event_ticker)
        event = event_data.get("event", {})
        print("Event Details:")
        print(f"Title: {safe_text(event.get('title'))}")
        print(f"Category: {safe_text(event.get('category'))}")

    if args.no_orderbook:
        return 0

    orderbook_data = get_orderbook(market_ticker)
    ob_fp = orderbook_data.get("orderbook_fp", {}) if isinstance(orderbook_data, dict) else {}
    yes_levels = top_orderbook_levels(ob_fp.get("yes_dollars"), args.orderbook_levels)
    no_levels = top_orderbook_levels(ob_fp.get("no_dollars"), args.orderbook_levels)
    print(f"\nOrderbook for {market_ticker}:")
    print("YES BIDS:")
    for price_dollars, qty_fp in yes_levels:
        print(f"  Price: ${price_dollars}, Quantity: {qty_fp}")
    print("\nNO BIDS:")
    for price_dollars, qty_fp in no_levels:
        print(f"  Price: ${price_dollars}, Quantity: {qty_fp}")
    print(f"\nBrowse: {MARKETS_UI_BASE}/{series_ticker.lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
