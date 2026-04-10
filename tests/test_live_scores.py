from deuce.live_scores import overlay_live_scores, parse_espn_scoreboard_html


SAMPLE_HTML = """
<section class="Card">
  <div class="Tournament_Header">
    <a class="AnchorLink Tournament_Link" href="/tennis/scoreboard/tournament/_/eventId/713-2026/date/20260323">
      Miami Open presented by Itau
    </a>
  </div>
  <div class="Grouping">
    <div class="Grouping_Name">Women's Singles</div>
    <div data-testid="match-cell">
      <div><div><span data-testid="live-indicator"></span>2nd Set</div></div>
      <div data-testid="competitor">
        <div>8</div>
        <a href="/tennis/player/_/id/9820/mirra-andreeva">Mirra Andreeva</a>
        <span>6</span><span>4</span><span>4</span>
      </div>
      <div data-testid="competitor">
        <div>10</div>
        <a href="/tennis/player/_/id/11219/victoria-mboko">Victoria Mboko</a>
        <span>7</span><span>7</span><span>3</span>
      </div>
      <div>Round 4 - Butch Buchholz</div>
    </div>
    <div data-testid="match-cell">
      <div><div>2:40 PM</div></div>
      <div data-testid="competitor">
        <a href="/tennis/player/_/id/3913/hailey-baptiste">Hailey Baptiste</a>
      </div>
      <div data-testid="competitor">
        <div>25</div>
        <a href="/tennis/player/_/id/2479/jelena-ostapenko">Jelena Ostapenko</a>
      </div>
      <div>Round 4 - Butch Buchholz</div>
    </div>
  </div>
</section>
"""


def test_parse_espn_scoreboard_html_live_and_scheduled():
    matches = parse_espn_scoreboard_html(SAMPLE_HTML, date_str="20260323")
    assert len(matches) == 2

    live_match = matches[0]
    assert live_match["is_live"] is True
    assert live_match["status_text"] == "2nd Set"
    assert live_match["round"] == "Round 4"
    assert live_match["court"] == "Butch Buchholz"
    assert live_match["players"][0]["name"] == "Mirra Andreeva"
    assert live_match["players"][0]["score_tokens"] == ["6", "4", "4"]

    scheduled = matches[1]
    assert scheduled["is_scheduled"] is True
    assert scheduled["status_text"] == "2:40 PM"
    assert scheduled["players"][1]["name"] == "Jelena Ostapenko"


def test_overlay_live_scores_matches_kalshi_matchup():
    matchups = [
        {
            "event": "Andreeva vs Mboko",
            "event_competition": "WTA Miami",
            "event_info_line": "WTA Miami · Round Of 16",
            "event_series_ticker": "KXWTAMATCH",
            "scheduled_start": "2026-03-23T21:20:00Z",
            "when": "2026-03-23T21:20:00Z",
            "market_phase": "pregame",
            "market_phase_label": "Pregame",
            "players": [
                {"name": "Mirra Andreeva"},
                {"name": "Victoria Mboko"},
            ],
        }
    ]
    scoreboard = parse_espn_scoreboard_html(SAMPLE_HTML, date_str="20260323")
    out = overlay_live_scores(matchups, scoreboard)
    assert out[0]["live_score"]["matched"] is True
    assert out[0]["market_phase"] == "live"
    assert "2nd Set" in out[0]["live_score"]["summary"]
