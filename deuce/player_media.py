"""Best-effort official player headshots from ATP / WTA profile pages."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import re
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from deuce.name_utils import normalize_player_name

ATP_RANKINGS_URL = "https://www.atptour.com/en/rankings/singles?rankRange=1-200"
WTA_RANKINGS_URL = "https://www.wtatennis.com/rankings/singles"
USER_AGENT = "Mozilla/5.0 (compatible; DEUCE/1.0)"


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess


def _slug_to_name(slug: str) -> str:
    return slug.replace("-", " ").strip().title()


def _tour_hint_from_matchup(matchup: Dict[str, Any]) -> str:
    blob = " ".join(
        [
            str(matchup.get("event_series_ticker") or ""),
            str(matchup.get("event_competition") or ""),
            str(matchup.get("event") or ""),
            str(matchup.get("event_info_line") or ""),
        ]
    ).lower()
    if "wta" in blob or "women" in blob:
        return "wta"
    return "atp"


@lru_cache(maxsize=1)
def _atp_index() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    html = _session().get(ATP_RANKINGS_URL, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href") or ""
        if "/en/players/" not in href or not href.endswith("/overview"):
            continue
        parts = [p for p in href.strip("/").split("/") if p]
        if len(parts) < 5:
            continue
        slug = parts[-3]
        code = parts[-2]
        display = _slug_to_name(slug)
        norm = normalize_player_name(display)
        if not norm:
            continue
        out[norm] = {
            "tour": "ATP",
            "profile_url": urljoin("https://www.atptour.com", href),
            "image_url": f"https://www.atptour.com/-/media/alias/player-headshot/{code}",
        }
    return out


@lru_cache(maxsize=1)
def _wta_index() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    html = _session().get(WTA_RANKINGS_URL, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href") or ""
        if not re.fullmatch(r"/players/\d+/[a-z0-9-]+", href):
            continue
        parts = [p for p in href.strip("/").split("/") if p]
        if len(parts) != 3:
            continue
        slug = parts[-1]
        display = _slug_to_name(slug)
        norm = normalize_player_name(display)
        if not norm or norm in out:
            continue
        out[norm] = {
            "tour": "WTA",
            "profile_url": urljoin("https://www.wtatennis.com", href),
        }
    return out


def extract_wta_headshot_url(html: str) -> Optional[str]:
    """Return the most player-specific WTA image we can find from a profile page."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        alt = str(img.get("alt") or "").lower()
        if "photoresources.wtatennis.com" not in src:
            continue
        if "home-share" in src.lower():
            continue
        if "torso" in alt or "player" in alt or "profile" in alt or "torso" in src.lower():
            candidates.append(src)
    if candidates:
        return candidates[0]

    meta = soup.find("meta", attrs={"property": "og:image"})
    src = meta.get("content") if meta else ""
    if isinstance(src, str) and "photoresources.wtatennis.com" in src and "home-share" not in src.lower():
        return src
    return None


@lru_cache(maxsize=256)
def lookup_player_media(player_name: str, tour_hint: str = "") -> Dict[str, str]:
    """Best-effort official player profile metadata."""
    norm = normalize_player_name(player_name)
    if not norm:
        return {}

    if tour_hint.lower() == "wta":
        record = _wta_index().get(norm)
        if record:
            out = dict(record)
            try:
                html = _session().get(record["profile_url"], timeout=20).text
                image_url = extract_wta_headshot_url(html)
                if image_url:
                    out["image_url"] = image_url
            except Exception:
                pass
            return out

    record = _atp_index().get(norm)
    if record:
        return dict(record)

    record = _wta_index().get(norm)
    if record:
        out = dict(record)
        try:
            html = _session().get(record["profile_url"], timeout=20).text
            image_url = extract_wta_headshot_url(html)
            if image_url:
                out["image_url"] = image_url
        except Exception:
            pass
        return out

    return {}


def enrich_matchups_with_player_media(matchups: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach official headshot/profile URLs when available."""
    enriched = [deepcopy(m) for m in matchups]
    for matchup in enriched:
        tour_hint = _tour_hint_from_matchup(matchup)
        for player in matchup.get("players") or []:
            info = lookup_player_media(str(player.get("name") or ""), tour_hint)
            if info.get("image_url"):
                player["image_url"] = info["image_url"]
            if info.get("profile_url"):
                player["profile_url"] = info["profile_url"]
            if info.get("tour"):
                player["tour"] = info["tour"]
    return enriched
