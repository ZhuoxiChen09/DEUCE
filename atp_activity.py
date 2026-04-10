"""
Fetch and parse ATP player activity pages to extract match results.
Returns matches in the format expected for training: player_a, player_b, winner (A or B).
Uses Playwright to load the JS-rendered page, then BeautifulSoup to parse.
"""
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

# Base URL for player activity (slug = e.g. "carlos-alcaraz/a0e2")
ATP_PLAYER_ACTIVITY_URL = (
    "https://www.atptour.com/en/players/{slug}/player-activity"
    "?matchType={match_type}&year={year}&tournament=all"
)


def _slug_to_display_name(slug: str) -> str:
    """Convert URL slug to display name: 'arthur-rinderknech' -> 'Arthur Rinderknech'."""
    if not slug:
        return ""
    # Handle 'alex-de-minaur' -> 'Alex de Minaur' (lowercase 'de' in middle)
    parts = slug.split("-")
    result = []
    for i, p in enumerate(parts):
        if p and p.lower() in ("de", "van", "von", "del", "la", "el") and i > 0 and i < len(parts) - 1:
            result.append(p.lower())
        else:
            result.append(p.capitalize() if p else "")
    return " ".join(result).strip()


def _format_score_display(score_text: str) -> str:
    """
    Convert ATP score string to human-readable set-by-set score, e.g. '6261' -> '6-2 6-1', '6736463' -> '6-7(3) 6-4 6-3'.
    """
    if not score_text:
        return ""
    s = score_text.strip()
    s = re.sub(r"\s*[Bb]ye\s*$", "", s).strip()
    if re.search(r"ret\.?|walkover|w/o", s, re.I):
        return s
    digits = re.sub(r"\D", "", s)
    if len(digits) < 2:
        return s
    parts = []
    i = 0
    while i < len(digits) - 1:
        a, b = int(digits[i]), int(digits[i + 1])
        i += 2
        if (a == 7 and b == 6) or (a == 6 and b == 7):
            tb = ""
            if i < len(digits):
                tb = f"({digits[i]})"
                i += 1
            parts.append(f"{a}-{b}{tb}")
        else:
            parts.append(f"{a}-{b}")
    return " ".join(parts)


def _parse_score_to_winner(score_text: str) -> Optional[bool]:
    """
    Parse score string to determine if the page player (first in each set) won.
    Score format: digit pairs per set, e.g. '6261' = 6-2 6-1 (page wins), '26626375' = 2-6 6-2 6-3 7-5 (page wins).
    Returns True if page player won, False if opponent won, None if unclear (e.g. Bye, walkover -> page wins by convention).
    """
    if not score_text:
        return None
    s = score_text.strip()
    # Strip trailing "Bye" (ATP sometimes appends it to score text)
    s = re.sub(r"\s*[Bb]ye\s*$", "", s).strip()
    # Retirement: on a player's activity page, "RET" usually means this player retired -> opponent wins
    if re.search(r"\bret\.?", s, re.I):
        return False
    # Walkover / w/o -> opponent didn't play, page player wins
    if re.search(r"walkover|w/o", s, re.I):
        return True
    # Score text "Bye" only (no digits) = no real match
    if re.sub(r"\D", "", s) == "":
        return None
    # Extract digits only; parse sets as (page, opp) with tiebreak skip after 7-6/6-7
    digits = re.sub(r"\D", "", s)
    if len(digits) < 2:
        return None
    page_sets = 0
    opp_sets = 0
    i = 0
    while i < len(digits) - 1:
        a, b = int(digits[i]), int(digits[i + 1])
        i += 2
        if a > b:
            page_sets += 1
        elif b > a:
            opp_sets += 1
        # After a 7-6 or 6-7 set, ATP often appends tiebreak games (e.g. 673 = 6-7(3)); skip one digit
        if ((a == 7 and b == 6) or (a == 6 and b == 7)) and i < len(digits):
            i += 1
    if page_sets > opp_sets:
        return True
    if opp_sets > page_sets:
        return False
    return None


def _get_player_name_from_page(soup: BeautifulSoup, slug: str) -> str:
    """Extract main player name from page (title or heading). Fallback: slug."""
    # <title>Carlos Alcaraz | Player Activity | ATP Tour | Tennis</title>
    title = soup.find("title")
    if title and title.get_text():
        t = title.get_text(strip=True)
        if "|" in t:
            return t.split("|")[0].strip()
    # First h1
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    # Use slug (e.g. "carlos-alcaraz/a0e2" -> take first part)
    base_slug = slug.split("/")[0] if "/" in slug else slug
    return _slug_to_display_name(base_slug)


def _parse_activity_html(html: str, page_slug: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse player activity HTML. Returns (page_player_name, list of match dicts).
    Each match: player_a (page player), player_b (opponent), winner (A|B), round, tournament, year.
    Page structure: opponent link and score link appear in the same row (tr or div); we find
    the smallest common parent of each score link that also contains an opponent link.
    """
    soup = BeautifulSoup(html, "html.parser")
    page_player = _get_player_name_from_page(soup, page_slug)
    matches: List[Dict[str, Any]] = []
    opp_re = re.compile(r"/en/players/[^/]+/[^/]+/overview")
    score_re = re.compile(r"/en/scores/match-stats/")

    score_links = soup.find_all("a", href=score_re)
    for a in score_links:
        score_text = (a.get_text() or "").strip()
        if not score_text or score_text.lower() == "bye":
            continue
        # Find smallest parent that contains both this score link and an opponent link
        parent = a.parent
        opp_link = None
        for _ in range(15):
            if parent is None:
                break
            candidates = parent.find_all("a", href=opp_re)
            for c in candidates:
                if c != a and c.get("href") and score_re.search(c.get("href", "")) is None:
                    opp_link = c
                    break
            if opp_link is not None:
                break
            parent = getattr(parent, "parent", None)

        if opp_link is None:
            continue
        opp_href = opp_link.get("href") or ""
        m = re.search(r"/en/players/([^/]+)/[^/]+/overview", opp_href)
        opp_slug = m.group(1) if m else ""
        opp_name = _slug_to_display_name(opp_slug) if opp_slug else (opp_link.get_text(strip=True) or "")
        if not opp_name:
            opp_name = _slug_to_display_name(opp_slug)
        round_text = ""
        round_pat = re.compile(r"^[RSQF]\d*$|^[QF]F$|^SF$|^F$|^BR\d*$|^RR$", re.I)
        for cell in parent.find_all(["td", "div", "span"]):
            t = (cell.get_text() or "").strip()
            if t and round_pat.match(t):
                round_text = t
                break
        if not round_text:
            for token in (parent.get_text() or "").split():
                t = token.strip()
                if t and round_pat.match(t):
                    round_text = t
                    break
        if not round_text:
            row = parent
            for _ in range(5):
                row = getattr(row, "parent", None)
                if not row:
                    break
                prev = row.find_previous_sibling()
                if prev:
                    t = (prev.get_text() or "").strip()
                    if round_pat.match(t):
                        round_text = t
                        break
                for child in getattr(row, "children", []):
                    if hasattr(child, "get_text"):
                        t = (child.get_text() or "").strip()
                        if t and round_pat.match(t):
                            round_text = t
                            break
                    if round_text:
                        break
                if round_text:
                    break
        tournament_name = ""
        year = ""
        section = parent
        tournament_link_re = re.compile(r"/en/tournaments/([^/]+)/\d+/overview")
        for _ in range(25):
            if section is None:
                break
            # Prefer tournament name from a preceding heading (skip "Header 2" etc.)
            prev = section.find_previous_sibling()
            if prev:
                for tag in ["h2", "h3", "h4"]:
                    h = prev.find(tag)
                    if h:
                        tname = (h.get_text() or "").strip()
                        if tname and tname not in ("Header 2", "Header 3", "2", "3") and not re.match(r"^\d+$", tname):
                            tournament_name = tname
                            break
            # Fallback: find link to tournament overview and use slug as name (e.g. doha -> Doha)
            if not tournament_name:
                tourn_a = section.find_previous("a", href=tournament_link_re)
                if tourn_a:
                    href = tourn_a.get("href") or ""
                    mo = tournament_link_re.search(href)
                    if mo:
                        slug = mo.group(1)
                        tournament_name = _slug_to_display_name(slug.replace("/", " "))
            if tournament_name:
                break
            section = getattr(section, "parent", None)
        score_display = _format_score_display(score_text)
        page_wins = _parse_score_to_winner(score_text)
        if page_wins is None:
            page_wins = True
        winner = "A" if page_wins else "B"
        matches.append({
            "player_a": page_player,
            "player_b": opp_name,
            "winner": winner,
            "score": score_display,
            "round": round_text,
            "tournament": tournament_name,
            "year": year,
        })
    return page_player, matches


def fetch_player_activity(
    player_slug: str,
    year: str = "all",
    match_type: str = "Singles",
    *,
    timeout_ms: int = 25000,
) -> List[Dict[str, Any]]:
    """
    Fetch match list for one player from ATP player activity page.
    player_slug: e.g. 'carlos-alcaraz/a0e2' or 'arthur-rinderknech/rc91'.
    year: 'all', '2026', '2025', etc.
    match_type: 'Singles' or 'Doubles'.
    Returns list of dicts with keys: player_a, player_b, winner (A|B), round, tournament, year.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required for ATP player activity. "
            "Install with: pip install playwright && playwright install chromium"
        ) from None

    url = ATP_PLAYER_ACTIVITY_URL.format(
        slug=player_slug.strip(),
        match_type=match_type.strip() or "Singles",
        year=year.strip() if year else "all",
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            page.wait_for_timeout(2000)
            html = page.content()
        finally:
            browser.close()

    _, matches = _parse_activity_html(html, player_slug)
    return matches


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "carlos-alcaraz/a0e2"
    year = sys.argv[2] if len(sys.argv) > 2 else "2026"
    print(f"Fetching activity for {slug} (year={year})...", file=sys.stderr)
    try:
        matches = fetch_player_activity(slug, year=year)
        print(f"Found {len(matches)} matches.\n", file=sys.stderr)
        for m in matches:
            winner_name = m["player_a"] if m.get("winner") == "A" else m["player_b"]
            loser_name = m["player_b"] if m.get("winner") == "A" else m["player_a"]
            score = m.get("score") or ""
            tournament = (m.get("tournament") or "").strip()
            round_ = (m.get("round") or "").strip()
            line = f"{winner_name} def. {loser_name}  {score}"
            if tournament or round_:
                extra = "  |  ".join(x for x in (tournament, round_) if x)
                line += f"  |  {extra}"
            print(line)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
