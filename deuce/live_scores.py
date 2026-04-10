"""Live tennis scoreboard routing with official-feed hooks and ESPN fallback."""

from __future__ import annotations

import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from bs4 import BeautifulSoup, Tag

from api import parse_iso8601_utc
from deuce.name_utils import normalize_player_name

ESPN_SCOREBOARD_URL = "https://www.espn.com/tennis/scoreboard/_/date/{date}"
ATP_PUBLIC_LIVE_URL = "https://app.atptour.com/api/v2/gateway/livematches/website?scoringTournamentLevel={level}"
ATP_PUBLIC_LIVE_LEVELS = ("tour", "challenger")
FLASHSCORE_TENNIS_URL = "https://www.flashscoreusa.com/tennis/"
MIAMI_OPEN_SCORES_URL = "https://prod.stats.miamiopen.com/api/v1/scores?tournament=0902&year={year}"
MIAMI_OPEN_UPDATES_URL = "https://prod.stats.miamiopen.com/api/v1/update?tournament=0902&year={year}"
WTA_PUBLIC_GLOBAL_MATCHES_URL = "https://api.wtatennis.com/tennis/matches/global"
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*[AP]M$", re.I)
_FLASHSCORE_FEED_RE = re.compile(r'cjs\.initialFeeds\["summary-results"\]\s*=\s*\{\s*data:\s*`(?P<data>.*?)`', re.S)
_FLASHSCORE_CACHE_TTL_S = 10
_FLASH_SCORE_URL_CACHE: Dict[str, Tuple[float, str]] = {}


@dataclass(frozen=True)
class LiveScoreProviderConfig:
    provider: str = "auto"
    official_base_url: str = ""
    official_api_key: str = ""
    official_date_param: str = "date"
    official_api_key_header: str = "Authorization"
    official_tour_param: str = "tour"
    official_timeout_s: int = 12
    public_miami_enabled: bool = True


def load_live_score_provider_config() -> LiveScoreProviderConfig:
    timeout_raw = os.getenv("DEUCE_LIVE_SCORE_TIMEOUT_S", "12").strip()
    try:
        timeout_s = max(3, int(timeout_raw))
    except ValueError:
        timeout_s = 12
    return LiveScoreProviderConfig(
        provider=os.getenv("DEUCE_LIVE_SCORE_PROVIDER", "auto").strip().lower() or "auto",
        official_base_url=os.getenv("DEUCE_OFFICIAL_LIVE_SCORE_URL", "").strip(),
        official_api_key=os.getenv("DEUCE_OFFICIAL_LIVE_SCORE_API_KEY", "").strip(),
        official_date_param=os.getenv("DEUCE_OFFICIAL_LIVE_SCORE_DATE_PARAM", "date").strip() or "date",
        official_api_key_header=os.getenv("DEUCE_OFFICIAL_LIVE_SCORE_API_KEY_HEADER", "Authorization").strip() or "Authorization",
        official_tour_param=os.getenv("DEUCE_OFFICIAL_LIVE_SCORE_TOUR_PARAM", "tour").strip() or "tour",
        official_timeout_s=timeout_s,
        public_miami_enabled=(os.getenv("DEUCE_PUBLIC_MIAMI_OPEN_ENABLED", "1").strip() != "0"),
    )


def _status_text_for_cell(cell: Tag) -> str:
    texts = [t.strip() for t in cell.stripped_strings if t and t.strip()]
    if not texts:
        return ""
    first = texts[0]
    if _TIME_RE.match(first) or "set" in first.lower() or first.lower() in ("final", "suspended", "postponed"):
        return first
    return ""


def _score_tokens_for_competitor(comp: Tag) -> List[str]:
    name_tag = comp.find("a")
    if name_tag is None:
        return []
    name = name_tag.get_text(" ", strip=True)
    texts = [t.strip() for t in comp.stripped_strings if t and t.strip()]
    if name not in texts:
        return []
    idx = texts.index(name)
    return [t for t in texts[idx + 1 :] if re.fullmatch(r"\d+", t)]


def _round_and_court(cell: Tag) -> tuple[str, str]:
    texts = [t.strip() for t in cell.stripped_strings if t and t.strip()]
    round_line = next((t for t in reversed(texts) if "Round" in t or "Final" in t or "Semifinal" in t or "Quarterfinal" in t), "")
    if " - " in round_line:
        round_text, court_text = round_line.split(" - ", 1)
        return round_text.strip(), court_text.strip()
    return round_line.strip(), ""


def parse_espn_scoreboard_html(html: str, *, date_str: str) -> List[Dict[str, Any]]:
    """Parse one ESPN tennis scoreboard page into structured singles matches."""
    soup = BeautifulSoup(html, "html.parser")
    out: List[Dict[str, Any]] = []

    for section in soup.find_all("section", class_="Card"):
        tournament_link = section.find("a", class_=re.compile(r"Tournament_Link"))
        tournament = tournament_link.get_text(" ", strip=True) if tournament_link else ""

        for grouping in section.find_all("div", class_=re.compile(r"Grouping")):
            name_node = grouping.find("div", class_=re.compile(r"Grouping_Name"))
            draw = name_node.get_text(" ", strip=True) if name_node else ""
            if draw not in ("Women's Singles", "Men's Singles"):
                continue

            for cell in grouping.find_all(attrs={"data-testid": "match-cell"}):
                competitors = cell.find_all(attrs={"data-testid": "competitor"})
                if len(competitors) < 2:
                    continue
                comp_rows: List[Dict[str, Any]] = []
                for comp in competitors[:2]:
                    name_tag = comp.find("a")
                    if name_tag is None:
                        continue
                    comp_rows.append(
                        {
                            "name": name_tag.get_text(" ", strip=True),
                            "score_tokens": _score_tokens_for_competitor(comp),
                        }
                    )
                if len(comp_rows) != 2:
                    continue

                status_text = _status_text_for_cell(cell)
                round_text, court_text = _round_and_court(cell)
                is_live = cell.find(attrs={"data-testid": "live-indicator"}) is not None
                status_lower = status_text.lower()
                is_completed = status_lower == "final"
                is_scheduled = bool(_TIME_RE.match(status_text))

                out.append(
                    {
                        "date": date_str,
                        "tournament": tournament,
                        "draw": draw,
                        "status_text": status_text,
                        "round": round_text,
                        "court": court_text,
                        "is_live": is_live,
                        "is_completed": is_completed,
                        "is_scheduled": is_scheduled,
                        "players": comp_rows,
                    }
                )
    return out


def _normalize_official_match(raw_match: Dict[str, Any], *, date_str: str) -> Optional[Dict[str, Any]]:
    players_raw = raw_match.get("players") or raw_match.get("competitors") or []
    players: List[Dict[str, Any]] = []
    for raw_player in players_raw[:2]:
        name = str(raw_player.get("name") or raw_player.get("player") or "").strip()
        if not name:
            continue
        score_tokens = raw_player.get("score_tokens") or raw_player.get("scores") or []
        if isinstance(score_tokens, str):
            score_tokens = [tok for tok in re.split(r"\s+", score_tokens.strip()) if tok]
        players.append({"name": name, "score_tokens": [str(tok) for tok in score_tokens]})
    if len(players) != 2:
        return None

    status_text = str(raw_match.get("status_text") or raw_match.get("status") or "").strip()
    state = str(raw_match.get("state") or "").strip().lower()
    is_live = bool(raw_match.get("is_live")) or state == "live"
    is_completed = bool(raw_match.get("is_completed")) or state in ("final", "completed")
    is_scheduled = bool(raw_match.get("is_scheduled")) or state == "scheduled" or bool(_TIME_RE.match(status_text))
    return {
        "date": str(raw_match.get("date") or date_str),
        "tournament": str(raw_match.get("tournament") or raw_match.get("competition") or ""),
        "draw": str(raw_match.get("draw") or raw_match.get("tour") or ""),
        "status_text": status_text,
        "round": str(raw_match.get("round") or ""),
        "court": str(raw_match.get("court") or ""),
        "is_live": is_live,
        "is_completed": is_completed,
        "is_scheduled": is_scheduled,
        "players": players,
        "source": str(raw_match.get("source") or "official"),
    }


def fetch_official_scoreboard_by_dates(
    date_strs: Sequence[str],
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_live_score_provider_config()
    if not cfg.official_base_url:
        return []

    out: List[Dict[str, Any]] = []
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    if cfg.official_api_key:
        headers[cfg.official_api_key_header] = cfg.official_api_key

    for ds in sorted(set(date_strs)):
        params = {cfg.official_date_param: ds}
        try:
            resp = session.get(cfg.official_base_url, params=params, headers=headers, timeout=cfg.official_timeout_s)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            continue
        matches_raw = payload.get("matches") if isinstance(payload, dict) else payload
        if not isinstance(matches_raw, list):
            continue
        for raw_match in matches_raw:
            if not isinstance(raw_match, dict):
                continue
            normalized = _normalize_official_match(raw_match, date_str=ds)
            if normalized is not None:
                out.append(normalized)
    return out


def _today_local_date_key() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d")


def _date_key_from_timestamp(ts: Any, *, fallback: str = "") -> str:
    dt = parse_iso8601_utc(ts)
    if dt is None:
        return fallback
    return dt.astimezone().strftime("%Y%m%d")


def _ordinal(n: int) -> str:
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _compact_name(first: Any, last: Any) -> str:
    return " ".join(part.strip() for part in (str(first or ""), str(last or "")) if str(part or "").strip()).strip()


def _score_token(score: Any, tiebreak: Any = "") -> Optional[str]:
    score_text = "" if score is None else str(score).strip()
    if not score_text:
        return None
    tiebreak_text = "" if tiebreak is None else str(tiebreak).strip()
    return f"{score_text}({tiebreak_text})" if tiebreak_text else score_text


def _wta_player_name(raw_match: Dict[str, Any], suffix: str) -> str:
    primary = _compact_name(raw_match.get(f"PlayerNameFirst{suffix}"), raw_match.get(f"PlayerNameLast{suffix}"))
    partner = _compact_name(raw_match.get(f"PlayerNameFirst{suffix}2"), raw_match.get(f"PlayerNameLast{suffix}2"))
    return f"{primary} / {partner}" if partner else primary


def _wta_player_row(raw_match: Dict[str, Any], suffix: str, *, include_points: bool) -> Optional[Dict[str, Any]]:
    name = _wta_player_name(raw_match, suffix)
    if not name:
        return None
    score_tokens: List[str] = []
    for set_num in range(1, 6):
        token = _score_token(raw_match.get(f"ScoreSet{set_num}{suffix}"), raw_match.get(f"ScoreTbSet{set_num}"))
        if token is not None:
            score_tokens.append(token)
    point = str(raw_match.get(f"Point{suffix}") or "").strip()
    if include_points and point:
        score_tokens.append(point)
    return {
        "name": name,
        "score_tokens": score_tokens,
        "country_code": str(raw_match.get(f"PlayerCountry{suffix}") or "").strip(),
        "player_id": str(raw_match.get(f"PlayerID{suffix}") or "").strip(),
    }


def _wta_round_text(raw_match: Dict[str, Any]) -> str:
    round_id = str(raw_match.get("RoundID") or "").strip()
    return f"Round {round_id}" if round_id else ""


def _wta_status_text(raw_match: Dict[str, Any], *, is_live: bool, is_completed: bool, is_scheduled: bool) -> str:
    if is_completed:
        return "Final"
    if is_scheduled:
        return str(raw_match.get("NotBefore") or raw_match.get("NotBeforeText") or "Scheduled").strip()
    if is_live:
        point_a = str(raw_match.get("PointA") or "").strip()
        point_b = str(raw_match.get("PointB") or "").strip()
        set_count = 0
        for set_num in range(1, 6):
            if str(raw_match.get(f"ScoreSet{set_num}A") or "").strip() or str(raw_match.get(f"ScoreSet{set_num}B") or "").strip():
                set_count = set_num
        if point_a or point_b:
            return f"{_ordinal(max(1, set_count))} Set"
        return "Live"
    return str(raw_match.get("ScoreString") or raw_match.get("ResultString") or raw_match.get("Message") or "").strip()


def _normalize_wta_public_match(raw_match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if str(raw_match.get("DrawMatchType") or "").upper() != "S":
        return None
    state = str(raw_match.get("MatchState") or "").strip().upper()
    point_a = str(raw_match.get("PointA") or "").strip()
    point_b = str(raw_match.get("PointB") or "").strip()
    is_completed = state == "F" or bool(raw_match.get("Winner"))
    is_scheduled = state in {"U", "S"} and not point_a and not point_b
    is_live = not is_completed and not is_scheduled and (
        state in {"P", "L", "R", "A", "W", "O"} or bool(point_a or point_b or raw_match.get("Serve"))
    )
    date_str = _date_key_from_timestamp(raw_match.get("MatchTimeStamp"), fallback=_today_local_date_key())
    players = [
        _wta_player_row(raw_match, "A", include_points=is_live),
        _wta_player_row(raw_match, "B", include_points=is_live),
    ]
    if any(player is None for player in players):
        return None
    tournament = raw_match.get("Tournament") if isinstance(raw_match.get("Tournament"), dict) else {}
    title = str(tournament.get("title") or "").strip()
    group = tournament.get("tournamentGroup") if isinstance(tournament.get("tournamentGroup"), dict) else {}
    return {
        "date": date_str,
        "tournament": title or str(group.get("name") or "").strip(),
        "draw": "WTA",
        "status_text": _wta_status_text(raw_match, is_live=is_live, is_completed=is_completed, is_scheduled=is_scheduled),
        "round": _wta_round_text(raw_match),
        "court": str(raw_match.get("CourtName") or "").strip(),
        "is_live": is_live,
        "is_completed": is_completed,
        "is_scheduled": is_scheduled,
        "players": [player for player in players if player is not None],
        "serve": str(raw_match.get("Serve") or "").strip(),
        "match_id": str(raw_match.get("MatchID") or "").strip(),
        "source": "wta-official-public",
    }


def _atp_status_text(raw_match: Dict[str, Any], *, is_live: bool, is_completed: bool, is_scheduled: bool) -> str:
    if is_completed:
        return "Final"
    if is_scheduled:
        return "Scheduled"
    if is_live:
        max_set = 0
        for team_key in ("PlayerTeam", "OpponentTeam"):
            for set_row in (raw_match.get(team_key) or {}).get("SetScores") or []:
                if set_row.get("SetScore") is not None:
                    max_set = max(max_set, int(set_row.get("SetNumber") or 0))
        return f"{_ordinal(max(1, max_set))} Set"
    return str(raw_match.get("RoundName") or "").strip()


def _atp_player_row(team: Dict[str, Any], *, include_game_score: bool) -> Optional[Dict[str, Any]]:
    player = team.get("Player") if isinstance(team, dict) else {}
    if not isinstance(player, dict):
        player = {}
    name = _compact_name(player.get("PlayerFirstName"), player.get("PlayerLastName"))
    if not name:
        return None
    score_tokens: List[str] = []
    for set_row in team.get("SetScores") or []:
        token = _score_token(set_row.get("SetScore"), set_row.get("TieBreakScore"))
        if token is not None:
            score_tokens.append(token)
    game_score = str(team.get("GameScore") or "").strip()
    if include_game_score and game_score:
        score_tokens.append(game_score)
    return {
        "name": name,
        "score_tokens": score_tokens,
        "country_code": str(player.get("PlayerCountry") or "").strip(),
        "player_id": str(player.get("PlayerId") or "").strip(),
    }


def _normalize_atp_public_match(raw_match: Dict[str, Any], *, tournament_name: str, date_str: str) -> Optional[Dict[str, Any]]:
    if bool(raw_match.get("IsDoubles")) or str(raw_match.get("Type") or "").lower() != "singles":
        return None
    status = str(raw_match.get("MatchStatus") or "").strip().upper()
    is_live = status == "P"
    is_completed = status in {"F", "C"} or bool(raw_match.get("WinningPlayerId"))
    is_scheduled = not is_live and not is_completed
    players = [
        _atp_player_row(raw_match.get("PlayerTeam") or {}, include_game_score=is_live),
        _atp_player_row(raw_match.get("OpponentTeam") or {}, include_game_score=is_live),
    ]
    if any(player is None for player in players):
        return None
    return {
        "date": date_str,
        "tournament": tournament_name,
        "draw": "ATP",
        "status_text": _atp_status_text(raw_match, is_live=is_live, is_completed=is_completed, is_scheduled=is_scheduled),
        "round": str(raw_match.get("RoundName") or "").strip(),
        "court": str(raw_match.get("CourtName") or "").strip(),
        "is_live": is_live,
        "is_completed": is_completed,
        "is_scheduled": is_scheduled,
        "players": [player for player in players if player is not None],
        "serve": str(raw_match.get("ServerTeam") or "").strip(),
        "match_id": str(raw_match.get("MatchId") or "").strip(),
        "source": "atp-official-public",
        "message": str(raw_match.get("ExtendedMessage") or raw_match.get("MatchStateReasonMessage") or "").strip(),
    }


def fetch_wta_public_scoreboard_by_dates(
    date_strs: Sequence[str],
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_live_score_provider_config()
    if not date_strs:
        return []
    try:
        resp = requests.get(
            WTA_PUBLIC_GLOBAL_MATCHES_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=cfg.official_timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    requested = set(date_strs)
    for raw_match in payload if isinstance(payload, list) else []:
        if not isinstance(raw_match, dict):
            continue
        normalized = _normalize_wta_public_match(raw_match)
        if normalized is not None and str(normalized.get("date") or "") in requested:
            out.append(normalized)
    return out


def fetch_atp_public_scoreboard_by_dates(
    date_strs: Sequence[str],
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_live_score_provider_config()
    today_key = _today_local_date_key()
    if not date_strs or today_key not in set(date_strs):
        return []

    session = requests.Session()
    out: List[Dict[str, Any]] = []
    for level in ATP_PUBLIC_LIVE_LEVELS:
        try:
            resp = session.get(
                ATP_PUBLIC_LIVE_URL.format(level=level),
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=cfg.official_timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            continue
        data = payload.get("Data") if isinstance(payload, dict) else {}
        tournaments = data.get("LiveMatchesTournamentsOrdered") if isinstance(data, dict) else []
        for tournament in tournaments if isinstance(tournaments, list) else []:
            if not isinstance(tournament, dict):
                continue
            tournament_name = str(tournament.get("EventTitle") or "").strip()
            for raw_match in tournament.get("LiveMatches") or []:
                if not isinstance(raw_match, dict):
                    continue
                normalized = _normalize_atp_public_match(raw_match, tournament_name=tournament_name, date_str=today_key)
                if normalized is not None:
                    out.append(normalized)
    return out


def fetch_espn_scoreboard_matches(
    date_str: str,
    *,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    sess = session or requests.Session()
    resp = sess.get(ESPN_SCOREBOARD_URL.format(date=date_str), timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return parse_espn_scoreboard_html(resp.text, date_str=date_str)


def fetch_espn_scoreboard_by_dates(date_strs: Sequence[str]) -> List[Dict[str, Any]]:
    session = requests.Session()
    out: List[Dict[str, Any]] = []
    for ds in sorted(set(date_strs)):
        try:
            out.extend(fetch_espn_scoreboard_matches(ds, session=session))
        except Exception:
            continue
    return out


def _normalize_miami_match(raw_match: Dict[str, Any], *, date_str: str, court_name: str = "") -> Optional[Dict[str, Any]]:
    teams = raw_match.get("teams") or []
    if len(teams) != 2:
        return None
    players: List[Dict[str, Any]] = []
    for team in teams[:2]:
        roster = team.get("players") or []
        if not roster:
            return None
        display_name = " / ".join(str(p.get("name") or "").strip() for p in roster if str(p.get("name") or "").strip())
        if not display_name:
            return None
        set_scores = []
        for set_row in team.get("sets") or []:
            score = str(set_row.get("score") or "").strip()
            tiebreak = str(set_row.get("tiebreak") or "").strip()
            if score:
                set_scores.append(f"{score}({tiebreak})" if tiebreak else score)
        points = str(team.get("points") or "").strip()
        score_tokens = list(set_scores)
        if points and points.lower() != "null":
            score_tokens.append(points)
        first_player = roster[0] if roster else {}
        players.append(
            {
                "name": display_name,
                "score_tokens": score_tokens,
                "image_url": str(first_player.get("image") or "").strip(),
                "country_code": str(first_player.get("country") or "").strip(),
                "player_id": str(first_player.get("playerId") or "").strip(),
            }
        )
    if len(players) != 2:
        return None

    status_text = str(raw_match.get("status") or "").strip()
    status_lower = status_text.lower()
    return {
        "date": date_str,
        "tournament": "Miami Open",
        "draw": str(raw_match.get("tour") or ""),
        "status_text": status_text,
        "round": str(raw_match.get("round") or ""),
        "court": court_name,
        "is_live": status_lower == "live",
        "is_completed": status_lower in ("finished", "walkover"),
        "is_scheduled": status_lower in ("upcoming", "suspended"),
        "players": players,
        "serve": str(raw_match.get("serve") or "").strip(),
        "match_id": str(raw_match.get("matchId") or "").strip(),
        "source": "miami-open-public",
    }


def fetch_miami_open_scoreboard_by_dates(
    date_strs: Sequence[str],
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_live_score_provider_config()
    if not cfg.public_miami_enabled or not date_strs:
        return []

    years = sorted({int(ds[:4]) for ds in date_strs if len(ds) >= 4 and ds[:4].isdigit()})
    session = requests.Session()
    out: List[Dict[str, Any]] = []

    for year in years:
        try:
            scores_resp = session.get(MIAMI_OPEN_SCORES_URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"}, timeout=cfg.official_timeout_s)
            scores_resp.raise_for_status()
            schedule_payload = scores_resp.json()
        except Exception:
            schedule_payload = []
        try:
            updates_resp = session.get(MIAMI_OPEN_UPDATES_URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"}, timeout=cfg.official_timeout_s)
            updates_resp.raise_for_status()
            updates_payload = updates_resp.json()
        except Exception:
            updates_payload = []

        by_match_id: Dict[str, Dict[str, Any]] = {}
        for day in schedule_payload if isinstance(schedule_payload, list) else []:
            date_str = str(day.get("date") or "").replace("-", "")
            if date_str not in date_strs:
                continue
            for court in day.get("courts") or []:
                court_name = str(court.get("name") or "")
                for raw_match in court.get("matches") or []:
                    normalized = _normalize_miami_match(raw_match, date_str=date_str, court_name=court_name)
                    if normalized is None:
                        continue
                    by_match_id[str(normalized.get("match_id") or normalized.get("summary") or id(normalized))] = normalized

        for raw_match in updates_payload if isinstance(updates_payload, list) else []:
            match_id = str(raw_match.get("matchId") or "").strip()
            if not match_id or match_id not in by_match_id:
                continue
            date_str = str(by_match_id[match_id].get("date") or "")
            court_name = str(by_match_id[match_id].get("court") or "")
            normalized = _normalize_miami_match(raw_match, date_str=date_str, court_name=court_name)
            if normalized is not None:
                by_match_id[match_id] = normalized

        out.extend(by_match_id.values())
    return out


def fetch_scoreboard_by_dates(
    date_strs: Sequence[str],
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> List[Dict[str, Any]]:
    cfg = config or load_live_score_provider_config()
    provider = cfg.provider
    if provider == "official":
        official = fetch_official_scoreboard_by_dates(date_strs, config=cfg)
        return official
    if provider == "officialpublic":
        public = fetch_wta_public_scoreboard_by_dates(date_strs, config=cfg)
        public.extend(fetch_atp_public_scoreboard_by_dates(date_strs, config=cfg))
        if public:
            return public
        return fetch_espn_scoreboard_by_dates(date_strs)
    if provider == "wtapublic":
        wta = fetch_wta_public_scoreboard_by_dates(date_strs, config=cfg)
        if wta:
            return wta
        return fetch_espn_scoreboard_by_dates(date_strs)
    if provider == "atppublic":
        atp = fetch_atp_public_scoreboard_by_dates(date_strs, config=cfg)
        if atp:
            return atp
        return fetch_espn_scoreboard_by_dates(date_strs)
    if provider == "miamiopen":
        miami = fetch_miami_open_scoreboard_by_dates(date_strs, config=cfg)
        if miami:
            return miami
        return fetch_espn_scoreboard_by_dates(date_strs)
    if provider == "espn":
        return fetch_espn_scoreboard_by_dates(date_strs)

    official = fetch_official_scoreboard_by_dates(date_strs, config=cfg)
    if official:
        return official
    wta_public = fetch_wta_public_scoreboard_by_dates(date_strs, config=cfg)
    atp_public = fetch_atp_public_scoreboard_by_dates(date_strs, config=cfg)
    public = [*wta_public, *atp_public]
    if public:
        return public
    miami = fetch_miami_open_scoreboard_by_dates(date_strs, config=cfg)
    if miami:
        return miami
    return fetch_espn_scoreboard_by_dates(date_strs)


def _local_date_key(ts: Any) -> str:
    dt = parse_iso8601_utc(ts)
    if dt is None:
        return ""
    return dt.astimezone().strftime("%Y%m%d")


def _tour_from_matchup(m: Dict[str, Any]) -> str:
    series = str(m.get("event_series_ticker") or "").upper()
    comp = str(m.get("event_competition") or "").upper()
    title = str(m.get("event") or "").upper()
    blob = " ".join([series, comp, title])
    if "WTA" in blob:
        return "wta"
    if "ATP" in blob:
        return "atp"
    return ""


def _tour_from_draw(draw: str) -> str:
    draw_lower = (draw or "").lower()
    if "women" in draw_lower or draw_lower == "wta":
        return "wta"
    if "men" in draw_lower or draw_lower == "atp":
        return "atp"
    return ""


def _name_signature(name: str) -> str:
    norm = normalize_player_name(name)
    if not norm:
        return ""
    parts = [p for p in norm.split(" ") if p]
    return " ".join(sorted(parts))


def _competition_token_blob(m: Dict[str, Any]) -> str:
    parts = [
        str(m.get("event_competition") or ""),
        str(m.get("event") or ""),
        str(m.get("event_info_line") or ""),
    ]
    return " ".join(parts).lower()


def _score_summary(match: Dict[str, Any]) -> str:
    left = " | ".join(
        f"{p['name']}: {' '.join(p.get('score_tokens') or ['-'])}"
        for p in match.get("players", [])
    )
    bits = [match.get("status_text") or "Status unknown", left]
    if match.get("round"):
        bits.append(match["round"])
    if match.get("court"):
        bits.append(match["court"])
    return " | ".join(bit for bit in bits if bit)


def _flashscore_slug(text: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return clean.strip("-")


def _flashscore_competition_text(matchup: Dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(matchup.get("event_competition") or "").strip(),
            str(matchup.get("event_info_line") or "").strip(),
            str(matchup.get("event") or "").strip(),
        )
        if part
    )


def _flashscore_tour_path(matchup: Dict[str, Any]) -> str:
    blob = _flashscore_competition_text(matchup).lower()
    if "challenger" in blob and "wta" in blob:
        return "challenger-women-singles"
    if "challenger" in blob:
        return "challenger-men-singles"
    if "wta" in blob:
        return "wta-singles"
    return "atp-singles"


def _flashscore_tournament_candidates(matchup: Dict[str, Any]) -> List[str]:
    competition = str(matchup.get("event_competition") or "").strip()
    cleaned = re.sub(r"\b(round|of|set|winner|match|singles|doubles)\b", " ", competition, flags=re.I)
    cleaned = re.sub(r"\b(atp|wta|challenger|masters|open)\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [tok for tok in re.split(r"[^A-Za-z0-9]+", cleaned) if tok]
    candidates: List[str] = []
    if tokens:
        candidates.append(_flashscore_slug(" ".join(tokens[-2:])))
        candidates.append(_flashscore_slug(tokens[-1]))
    event_blob = str(matchup.get("event") or "").strip()
    for tok in re.split(r"[^A-Za-z0-9]+", event_blob):
        if tok and len(tok) > 3:
            candidates.append(_flashscore_slug(tok))
    out: List[str] = []
    seen = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _fetch_text_with_cache(url: str, *, timeout_s: int) -> str:
    now = time.time()
    cached = _FLASH_SCORE_URL_CACHE.get(url)
    if cached and now - cached[0] <= _FLASHSCORE_CACHE_TTL_S:
        return cached[1]
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout_s)
    resp.raise_for_status()
    text = resp.text
    _FLASH_SCORE_URL_CACHE[url] = (now, text)
    return text


def _extract_flashscore_feed(html: str) -> str:
    match = _FLASHSCORE_FEED_RE.search(html or "")
    return match.group("data") if match else ""


def _parse_flashscore_event_chunk(chunk: str, *, tournament: str, draw: str) -> Optional[Dict[str, Any]]:
    if not chunk or "?" not in chunk:
        return None
    event_id, rest = chunk.split("?", 1)
    parts = rest.split("?")
    fields: Dict[str, str] = {"AA": event_id}
    for idx in range(0, len(parts) - 1, 2):
        key = parts[idx].strip()
        value = parts[idx + 1]
        if key:
            fields[key] = value

    first_name = str(fields.get("AE") or "").strip()
    second_name = str(fields.get("AF") or "").strip()
    if not first_name or not second_name:
        return None

    first_scores = [str(fields.get(code) or "").strip() for code in ("BA", "BC", "BE", "BG", "BI")]
    second_scores = [str(fields.get(code) or "").strip() for code in ("BB", "BD", "BF", "BH", "BJ")]
    first_tokens = [score for score in first_scores if score]
    second_tokens = [score for score in second_scores if score]

    start_ts = fields.get("AD")
    date_str = ""
    if start_ts and str(start_ts).isdigit():
        dt = datetime.fromtimestamp(int(str(start_ts)), tz=datetime.now().astimezone().tzinfo)
        date_str = dt.strftime("%Y%m%d")

    first_sets_won = sum(
        1
        for a, b in zip(first_scores, second_scores)
        if a and b and a.isdigit() and b.isdigit() and int(a) > int(b)
    )
    second_sets_won = sum(
        1
        for a, b in zip(first_scores, second_scores)
        if a and b and a.isdigit() and b.isdigit() and int(b) > int(a)
    )
    completed = max(first_sets_won, second_sets_won) >= 2 or str(fields.get("ER") or "").strip().lower() == "final"
    live = bool(first_tokens or second_tokens) and not completed

    return {
        "date": date_str,
        "tournament": tournament,
        "draw": draw,
        "status_text": "Final" if completed else ("Live" if live else ""),
        "round": str(fields.get("ER") or "").strip(),
        "court": "",
        "is_live": live,
        "is_completed": completed,
        "is_scheduled": False,
        "players": [
            {
                "name": first_name,
                "score_tokens": first_tokens,
                "image_url": str(fields.get("OA") or "").strip(),
                "country_code": str(fields.get("CA") or "").strip(),
                "player_id": str(fields.get("PX") or "").strip(),
            },
            {
                "name": second_name,
                "score_tokens": second_tokens,
                "image_url": str(fields.get("OB") or "").strip(),
                "country_code": str(fields.get("CB") or "").strip(),
                "player_id": str(fields.get("PY") or "").strip(),
            },
        ],
        "source": "flashscore-public-html",
        "match_url": (
            f"https://www.flashscoreusa.com/game/tennis/"
            f"{str(fields.get('WU') or '').strip()}-{str(fields.get('PX') or '').strip()}/"
            f"{str(fields.get('WV') or '').strip()}-{str(fields.get('PY') or '').strip()}/"
        ),
    }


def _parse_flashscore_tournament_page(html: str, *, default_tournament: str = "", draw: str = "") -> List[Dict[str, Any]]:
    feed = _extract_flashscore_feed(html)
    if not feed:
        return []
    chunks = feed.split("~AA?")
    header = chunks[0]
    tournament = default_tournament
    if not tournament:
        header_match = re.search(r"~ZA\?([^~]+)", header)
        if header_match:
            tournament = header_match.group(1).strip()
    out: List[Dict[str, Any]] = []
    for chunk in chunks[1:]:
        normalized = _parse_flashscore_event_chunk(chunk, tournament=tournament, draw=draw)
        if normalized is not None:
            out.append(normalized)
    return out


def _fetch_flashscore_match_for_matchup(
    matchup: Dict[str, Any],
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> Optional[Dict[str, Any]]:
    cfg = config or load_live_score_provider_config()
    path = _flashscore_tour_path(matchup)
    draw = "WTA" if "wta" in path else "ATP"
    competition_blob = _flashscore_competition_text(matchup).lower()
    for slug in _flashscore_tournament_candidates(matchup):
        url = f"https://www.flashscoreusa.com/tennis/{path}/{slug}/"
        try:
            html = _fetch_text_with_cache(url, timeout_s=cfg.official_timeout_s)
        except Exception:
            continue
        parsed = _parse_flashscore_tournament_page(html, default_tournament=str(matchup.get("event_competition") or "").strip(), draw=draw)
        pair = sorted(_name_signature(p.get("name", "")) for p in matchup.get("players", []))
        candidates: List[Dict[str, Any]] = []
        for match in parsed:
            score_pair = sorted(_name_signature(p.get("name", "")) for p in match.get("players", []))
            if score_pair != pair:
                continue
            rank = 1 if slug and slug in competition_blob else 0
            candidate = dict(match)
            candidate["_rank"] = rank
            candidates.append(candidate)
        if candidates:
            return sorted(candidates, key=lambda item: (item.get("is_completed", False), item.get("is_live", False), item.get("_rank", 0)), reverse=True)[0]
    return None


def _choose_best_live_score(*scores: Dict[str, Any]) -> Dict[str, Any]:
    ranked: List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = []
    source_rank = {
        "official": 5,
        "wta-official-public": 4,
        "atp-official-public": 4,
        "miami-open-public": 3,
        "flashscore-public-html": 2,
        "espn": 1,
    }
    for score in scores:
        if not isinstance(score, dict) or not score:
            continue
        ranked.append(
            (
                (
                    1 if score.get("matched") else 0,
                    1 if score.get("is_completed") else 0,
                    1 if score.get("is_live") else 0,
                    source_rank.get(str(score.get("source") or ""), 0),
                ),
                score,
            )
        )
    if not ranked:
        return {"matched": False}
    return sorted(ranked, key=lambda item: item[0], reverse=True)[0][1]


def resolve_live_score_for_matchup(
    matchup: Dict[str, Any],
    *,
    scoreboard_matches: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[LiveScoreProviderConfig] = None,
) -> Dict[str, Any]:
    enriched = overlay_live_scores([matchup], scoreboard_matches=scoreboard_matches, config=config)
    live_score = (enriched[0].get("live_score") if enriched else None) or {}
    flashscore = _fetch_flashscore_match_for_matchup(matchup, config=config)
    if flashscore:
        flashscore = {
            "source": str(flashscore.get("source") or "flashscore-public-html"),
            "matched": True,
            "status_text": flashscore.get("status_text", ""),
            "is_live": bool(flashscore.get("is_live")),
            "is_completed": bool(flashscore.get("is_completed")),
            "is_scheduled": bool(flashscore.get("is_scheduled")),
            "tournament": flashscore.get("tournament", ""),
            "draw": flashscore.get("draw", ""),
            "round": flashscore.get("round", ""),
            "court": flashscore.get("court", ""),
            "players": flashscore.get("players", []),
            "summary": _score_summary(flashscore),
        }
    if live_score or flashscore:
        return _choose_best_live_score(live_score, flashscore)
    existing = matchup.get("live_score")
    if isinstance(existing, dict) and existing:
        return deepcopy(existing)
    return {"matched": False}


def overlay_live_scores(
    matchups: Sequence[Dict[str, Any]],
    scoreboard_matches: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    config: Optional[LiveScoreProviderConfig] = None,
) -> List[Dict[str, Any]]:
    """Attach best-effort live score data to Kalshi matchup rows."""
    enriched = [deepcopy(m) for m in matchups]
    date_keys = [_local_date_key(m.get("scheduled_start") or m.get("when")) for m in enriched]
    scoreboard = list(scoreboard_matches) if scoreboard_matches is not None else fetch_scoreboard_by_dates([d for d in date_keys if d], config=config)

    for matchup in enriched:
        players = matchup.get("players") or []
        if len(players) != 2:
            continue
        local_date = _local_date_key(matchup.get("scheduled_start") or matchup.get("when"))
        if not local_date:
            continue
        pair = sorted(_name_signature(p.get("name", "")) for p in players)
        matchup_tour = _tour_from_matchup(matchup)
        competition_blob = _competition_token_blob(matchup)

        candidates: List[Dict[str, Any]] = []
        for score_match in scoreboard:
            if score_match.get("date") != local_date:
                continue
            score_pair = sorted(_name_signature(p.get("name", "")) for p in score_match.get("players", []))
            if score_pair != pair:
                continue
            score_tour = _tour_from_draw(str(score_match.get("draw") or ""))
            if matchup_tour and score_tour and matchup_tour != score_tour:
                continue
            score_tournament = str(score_match.get("tournament") or "").lower()
            score_rank = 1 if score_tournament and any(tok in score_tournament for tok in ("miami", "indian wells", "open")) and "miami" in competition_blob else 0
            candidate = dict(score_match)
            candidate["_rank"] = score_rank
            candidates.append(candidate)

        if not candidates:
            continue
        best = sorted(candidates, key=lambda x: (x.get("_rank", 0), x.get("is_live", False), x.get("is_completed", False)), reverse=True)[0]
        source = str(best.get("source") or ("official" if best.get("source") else "espn"))

        matchup["live_score"] = {
            "source": source,
            "matched": True,
            "status_text": best.get("status_text", ""),
            "is_live": bool(best.get("is_live")),
            "is_completed": bool(best.get("is_completed")),
            "is_scheduled": bool(best.get("is_scheduled")),
            "tournament": best.get("tournament", ""),
            "draw": best.get("draw", ""),
            "round": best.get("round", ""),
            "court": best.get("court", ""),
            "players": best.get("players", []),
            "summary": _score_summary(best),
        }

        if best.get("is_live"):
            matchup["market_phase"] = "live"
            matchup["market_phase_label"] = "Live"
            matchup["market_phase_detail"] = f"Live scoreboard matched from {source.upper()}."
        elif best.get("is_completed"):
            matchup["market_phase"] = "completed"
            matchup["market_phase_label"] = "Final"
            matchup["market_phase_detail"] = f"Completed match matched from {source.upper()}."
        elif best.get("is_scheduled"):
            matchup["market_phase"] = "pregame"
            matchup["market_phase_label"] = "Pregame"
            matchup["market_phase_detail"] = f"Scheduled match matched from {source.upper()}."
    return enriched
