"""
Fetch ATP match results from ESPN scoreboard by date.
Uses requests + BeautifulSoup (no Playwright). Men's Singles only for training data.
Tournament catalog (dates) is maintained separately (e.g. data/tournament_catalog.json); we do not pull schedule from ESPN.
"""
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup

ESPN_SCOREBOARD_URL = "https://www.espn.com/tennis/scoreboard/_/date/{date}"
# Date format YYYYMMDD
PLAYER_LINK_RE = re.compile(r"/tennis/player/_/id/\d+/([^/]+)")


def _expand_range(start: datetime, end: datetime) -> Set[str]:
    """Return set of YYYYMMDD strings from start to end inclusive."""
    out = set()
    d = start
    while d <= end:
        out.add(d.strftime("%Y%m%d"))
        d = d + timedelta(days=1)
    return out


def get_dates_with_matches(
    season: int,
    *,
    catalog_path: Optional[str] = None,
) -> List[str]:
    """
    Return sorted list of dates (YYYYMMDD) that fall within any tournament date range for the season.
    Reads from tournament catalog JSON (no ESPN schedule fetch). Use these dates to call get_scoreboard_matches(date).
    """
    try:
        from deuce.tournament_catalog import load_catalog
    except ImportError:
        return []
    catalog = load_catalog(catalog_path) if catalog_path else load_catalog()
    all_dates: Set[str] = set()
    for entry in catalog:
        if entry.get("season") != season:
            continue
        start_s, end_s = entry.get("date_start") or "", entry.get("date_end") or ""
        if start_s and end_s and len(start_s) == 8 and len(end_s) == 8:
            try:
                start_dt = datetime(int(start_s[:4]), int(start_s[4:6]), int(start_s[6:8]))
                end_dt = datetime(int(end_s[:4]), int(end_s[4:6]), int(end_s[6:8]))
                all_dates |= _expand_range(start_dt, end_dt)
            except (ValueError, TypeError):
                pass
    return sorted(all_dates)


def _count_sets_from_score(score_text: str) -> Tuple[int, int]:
    """
    Return (sets_player_a, sets_player_b) from one player's score string (player_a's games per set).
    Handles 6-6 (tiebreak: skip next digit, set to player_a) and 7-6/6-7 (skip tiebreak digit).
    """
    digits = re.sub(r"\D", "", score_text or "")
    if len(digits) < 2:
        return (0, 0)
    sets_a, sets_b = 0, 0
    i = 0
    while i < len(digits) - 1:
        a, b = int(digits[i]), int(digits[i + 1])
        i += 2
        if a == 6 and b == 6:
            if i < len(digits):
                i += 1
            sets_a += 1
        elif (a == 7 and b == 6) or (a == 6 and b == 7):
            if i < len(digits):
                i += 1
            if a > b:
                sets_a += 1
            else:
                sets_b += 1
        elif a > b:
            sets_a += 1
        elif b > a:
            sets_b += 1
    return (sets_a, sets_b)


def _format_score_display(score_text: str) -> str:
    """Human-readable set score from digit string (reuse logic from atp_activity)."""
    digits = re.sub(r"\D", "", score_text or "")
    if len(digits) < 2:
        return score_text or ""
    parts = []
    i = 0
    while i < len(digits) - 1:
        a, b = int(digits[i]), int(digits[i + 1])
        i += 2
        if (a == 7 and b == 6) or (a == 6 and b == 7):
            tb = f"({digits[i]})" if i < len(digits) else ""
            if i < len(digits):
                i += 1
            parts.append(f"{a}-{b}{tb}")
        else:
            parts.append(f"{a}-{b}")
    return " ".join(parts)


def get_scoreboard_matches(
    date_str: str,
    atp_only: bool = True,
    *,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch ESPN scoreboard for date (YYYYMMDD) and return list of match dicts.
    atp_only: if True, only Men's Singles. Each match: player_a, player_b, winner (A|B), score, tournament, round, date.
    """
    url = ESPN_SCOREBOARD_URL.format(date=date_str)
    sess = session or requests.Session()
    try:
        r = sess.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (compatible; TennisBot/1.0)"})
        r.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    matches: List[Dict[str, Any]] = []
    tournament_name = ""
    section_men = False
    player_link_re = re.compile(r"/tennis/player/_/id/\d+/([^/]+)")

    # Find tournament link at top (e.g. BNP Paribas Open)
    tourn_a = soup.find("a", href=re.compile(r"/tennis/scoreboard/tournament/"))
    if tourn_a and tourn_a.get_text(strip=True):
        tournament_name = tourn_a.get_text(strip=True)

    # Collect all links to players and all digit-only score-like text
    all_links = soup.find_all("a", href=player_link_re)
    # We need to associate player names with scores. Page structure: sections "Women's Singles", "Men's Singles";
    # then for each match: status, optional seed, player1, score1, player2, score2, round.
    # So we scan for "Men's Singles" and then take subsequent content until next section.
    text = soup.get_text()
    if atp_only and "Men's Singles" not in text:
        return []

    # Collect (player_name, score_str) pairs in document order within Men's Singles only.
    pairs: List[Tuple[str, str]] = []
    for link in soup.find_all("a", href=player_link_re):
        parent = link.parent
        in_men = False
        for _ in range(15):
            if parent is None:
                break
            ptext = (parent.get_text() or "") if hasattr(parent, "get_text") else ""
            if "Women's Singles" in ptext:
                wpos = ptext.find("Women's Singles")
                if wpos >= 0 and "Men's Singles" not in ptext[max(0, wpos - 50): wpos + 30]:
                    break
            if "Men's Singles" in ptext:
                in_men = True
                break
            parent = getattr(parent, "parent", None)
        if not in_men:
            continue
        name = link.get_text(strip=True) or (player_link_re.search(link.get("href") or "").group(1).replace("-", " ").title() if player_link_re.search(link.get("href") or "") else "")
        if not name:
            continue
        next_score = None
        for elem in link.find_next_siblings():
            t = (elem.get_text() or "").strip() if hasattr(elem, "get_text") else ""
            digits = re.sub(r"\D", "", t)
            if len(digits) >= 2 and digits.isdigit():
                next_score = digits
                break
        if not next_score:
            continue
        pairs.append((name, next_score))

    # Group into matches: (p1, s1), (p2, s2) -> one match
    for i in range(0, len(pairs) - 1, 2):
        (name_a, score_a), (name_b, score_b) = pairs[i], pairs[i + 1]
        sets_a, sets_b = _count_sets_from_score(score_a)
        winner = "A" if sets_a > sets_b else "B"
        score_display = _format_score_display(score_a) + " / " + _format_score_display(score_b)
        matches.append({
            "player_a": name_a,
            "player_b": name_b,
            "winner": winner,
            "score": score_display,
            "tournament": tournament_name,
            "round": "",
            "date": date_str,
        })
    return matches


def fetch_espn_matches_for_season(
    season: int,
    atp_only: bool = True,
    *,
    max_dates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Get all dates from schedule for season, then fetch scoreboard for each date and collect matches.
    Optionally limit number of dates to avoid too many requests (max_dates).
    """
    dates = get_dates_with_matches(season)
    if max_dates is not None:
        dates = dates[:max_dates]
    all_matches = []
    sess = requests.Session()
    for d in dates:
        ms = get_scoreboard_matches(d, atp_only=atp_only, session=sess)
        all_matches.extend(ms)
    # Dedupe by (date, player_a, player_b)
    seen = set()
    out = []
    for m in all_matches:
        key = (m["date"], tuple(sorted([m["player_a"], m["player_b"]])))
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out
