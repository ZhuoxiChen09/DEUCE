from deuce.live_scores import (
    ATP_PUBLIC_LIVE_LEVELS,
    LiveScoreProviderConfig,
    _parse_flashscore_tournament_page,
    _normalize_atp_public_match,
    _normalize_wta_public_match,
    _normalize_miami_match,
    _normalize_official_match,
    overlay_live_scores,
    resolve_live_score_for_matchup,
)


def test_normalize_official_match_supports_generic_json_payload():
    raw = {
        "date": "20260323",
        "tournament": "Miami Open",
        "draw": "WTA",
        "status": "1st Set",
        "state": "live",
        "round": "Round 4",
        "court": "Stadium",
        "players": [
            {"name": "Elena Rybakina", "scores": ["6", "0", "0"]},
            {"name": "Talia Gibson", "scores": ["2", "1", "15"]},
        ],
        "source": "official-demo",
    }
    out = _normalize_official_match(raw, date_str="20260323")
    assert out is not None
    assert out["is_live"] is True
    assert out["players"][0]["score_tokens"] == ["6", "0", "0"]
    assert out["source"] == "official-demo"


def test_overlay_live_scores_preserves_official_source_label():
    matchups = [
        {
            "event": "Rybakina vs Gibson",
            "event_competition": "WTA Miami",
            "event_info_line": "WTA Miami | Round Of 16",
            "event_series_ticker": "KXWTAMATCH",
            "scheduled_start": "2026-03-23T17:00:00Z",
            "when": "2026-03-23T17:00:00Z",
            "players": [{"name": "Elena Rybakina"}, {"name": "Talia Gibson"}],
        }
    ]
    scoreboard = [
        {
            "date": "20260323",
            "tournament": "Miami Open",
            "draw": "WTA",
            "status_text": "1st Set",
            "round": "Round 4",
            "court": "Stadium",
            "is_live": True,
            "is_completed": False,
            "is_scheduled": False,
            "players": [
                {"name": "Elena Rybakina", "score_tokens": ["6", "0", "0"]},
                {"name": "Talia Gibson", "score_tokens": ["2", "1", "15"]},
            ],
            "source": "official-demo",
        }
    ]
    out = overlay_live_scores(matchups, scoreboard_matches=scoreboard, config=LiveScoreProviderConfig(provider="official"))
    assert out[0]["live_score"]["source"] == "official-demo"
    assert out[0]["market_phase"] == "live"


def test_normalize_miami_match_keeps_points_and_tiebreaks():
    raw = {
        "matchId": "LD006",
        "tour": "WTA",
        "status": "Live",
        "round": "Round 4",
        "serve": "2",
        "teams": [
            {
                "players": [{"playerId": "1", "name": "Elena Rybakina", "country": "KAZ", "image": "https://img/rybakina.png"}],
                "sets": [{"score": "6"}, {"score": "0"}],
                "points": "0",
            },
            {
                "players": [{"playerId": "2", "name": "Talia Gibson", "country": "AUS", "image": "https://img/gibson.png"}],
                "sets": [{"score": "2"}, {"score": "1", "tiebreak": "3"}],
                "points": "15",
            },
        ],
    }
    out = _normalize_miami_match(raw, date_str="20260324", court_name="Stadium")
    assert out is not None
    assert out["players"][0]["score_tokens"] == ["6", "0", "0"]
    assert out["players"][1]["score_tokens"] == ["2", "1(3)", "15"]
    assert out["source"] == "miami-open-public"


def test_normalize_wta_public_match_keeps_subscores_and_points():
    raw = {
        "DrawMatchType": "S",
        "MatchState": "P",
        "MatchTimeStamp": "2026-03-24T14:00:00-04:00",
        "MatchID": "LS005",
        "CourtName": "Stadium",
        "RoundID": 4,
        "PlayerNameFirstA": "Elena",
        "PlayerNameLastA": "Rybakina",
        "PlayerCountryA": "KAZ",
        "PlayerIDA": "320760",
        "PlayerNameFirstB": "Talia",
        "PlayerNameLastB": "Gibson",
        "PlayerCountryB": "AUS",
        "PlayerIDB": "329999",
        "ScoreSet1A": "6",
        "ScoreSet1B": "2",
        "ScoreSet2A": "0",
        "ScoreSet2B": "1",
        "PointA": "0",
        "PointB": "15",
        "Serve": "2",
        "Tournament": {
            "title": "Miami Open presented by Itau - Miami, FL, USA",
            "tournamentGroup": {"name": "MIAMI"},
        },
    }
    out = _normalize_wta_public_match(raw)
    assert out is not None
    assert out["is_live"] is True
    assert out["players"][0]["score_tokens"] == ["6", "0", "0"]
    assert out["players"][1]["score_tokens"] == ["2", "1", "15"]
    assert out["source"] == "wta-official-public"


def test_normalize_atp_public_match_keeps_subscores_and_game_points():
    raw = {
        "Type": "singles",
        "IsDoubles": False,
        "MatchStatus": "P",
        "MatchId": "MS022",
        "RoundName": "Round of 32",
        "CourtName": "Estadio Tres Marias",
        "ServerTeam": 0,
        "ExtendedMessage": "Marc Polmans wins the point on his 2nd serve.",
        "PlayerTeam": {
            "Player": {
                "PlayerId": "PG94",
                "PlayerCountry": "AUS",
                "PlayerFirstName": "Marc",
                "PlayerLastName": "Polmans",
            },
            "GameScore": "30",
            "SetScores": [
                {"SetNumber": 1, "SetScore": 6, "TieBreakScore": None},
                {"SetNumber": 2, "SetScore": 3, "TieBreakScore": None},
                {"SetNumber": 3, "SetScore": 0, "TieBreakScore": None},
            ],
        },
        "OpponentTeam": {
            "Player": {
                "PlayerId": "MJ98",
                "PlayerCountry": "ARG",
                "PlayerFirstName": "Facundo",
                "PlayerLastName": "Mena",
            },
            "GameScore": "15",
            "SetScores": [
                {"SetNumber": 1, "SetScore": 3, "TieBreakScore": None},
                {"SetNumber": 2, "SetScore": 6, "TieBreakScore": None},
                {"SetNumber": 3, "SetScore": 0, "TieBreakScore": None},
            ],
        },
    }
    out = _normalize_atp_public_match(raw, tournament_name="Morelia Open", date_str="20260324")
    assert out is not None
    assert out["is_live"] is True
    assert out["players"][0]["score_tokens"] == ["6", "3", "0", "30"]
    assert out["players"][1]["score_tokens"] == ["3", "6", "0", "15"]
    assert out["source"] == "atp-official-public"


def test_fetch_atp_public_scoreboard_requests_tour_and_challenger(monkeypatch):
    calls = []

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeSession:
        def get(self, url, headers=None, timeout=None):
            calls.append(url)
            if "scoringTournamentLevel=tour" in url:
                return _FakeResponse(
                    {
                        "Data": {
                            "LiveMatchesTournamentsOrdered": [
                                {
                                    "EventTitle": "Miami Open presented by Itau",
                                    "LiveMatches": [
                                        {
                                            "Type": "singles",
                                            "IsDoubles": False,
                                            "MatchStatus": "P",
                                            "MatchId": "MS100",
                                            "RoundName": "Final",
                                            "CourtName": "Stadium",
                                            "PlayerTeam": {
                                                "Player": {
                                                    "PlayerId": "A1",
                                                    "PlayerCountry": "ITA",
                                                    "PlayerFirstName": "Jannik",
                                                    "PlayerLastName": "Sinner",
                                                },
                                                "GameScore": "30",
                                                "SetScores": [{"SetNumber": 1, "SetScore": 6, "TieBreakScore": None}],
                                            },
                                            "OpponentTeam": {
                                                "Player": {
                                                    "PlayerId": "A2",
                                                    "PlayerCountry": "ESP",
                                                    "PlayerFirstName": "Carlos",
                                                    "PlayerLastName": "Alcaraz",
                                                },
                                                "GameScore": "15",
                                                "SetScores": [{"SetNumber": 1, "SetScore": 4, "TieBreakScore": None}],
                                            },
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                )
            return _FakeResponse({"Data": {"LiveMatchesTournamentsOrdered": []}})

    monkeypatch.setattr("deuce.live_scores.requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("deuce.live_scores._today_local_date_key", lambda: "20260324")

    from deuce.live_scores import fetch_atp_public_scoreboard_by_dates

    rows = fetch_atp_public_scoreboard_by_dates(["20260324"])
    assert len(rows) == 1
    assert rows[0]["tournament"] == "Miami Open presented by Itau"
    assert any("scoringTournamentLevel=tour" in url for url in calls)
    assert any("scoringTournamentLevel=challenger" in url for url in calls)
    assert ATP_PUBLIC_LIVE_LEVELS == ("tour", "challenger")


def test_resolve_live_score_for_matchup_uses_fresh_scoreboard_match():
    matchup = {
        "event": "Kozlov vs Martin",
        "event_competition": "ATP Challenger Morelia",
        "event_info_line": "ATP Challenger Morelia | Round Of 32",
        "event_series_ticker": "KXATPMATCH",
        "scheduled_start": "2026-03-24T20:00:00Z",
        "when": "2026-03-24T20:00:00Z",
        "players": [{"name": "Stefan Kozlov"}, {"name": "Andres Martin"}],
        "live_score": {"matched": False},
    }
    scoreboard = [
        {
            "date": "20260324",
            "tournament": "Morelia Open",
            "draw": "ATP",
            "status_text": "Final",
            "round": "Round of 32",
            "court": "Grandstand Pablo Arroyo",
            "is_live": False,
            "is_completed": True,
            "is_scheduled": False,
            "players": [
                {"name": "Stefan Kozlov", "score_tokens": ["2", "2"]},
                {"name": "Andres Martin", "score_tokens": ["6", "6"]},
            ],
            "source": "atp-official-public",
        }
    ]
    out = resolve_live_score_for_matchup(matchup, scoreboard_matches=scoreboard)
    assert out["matched"] is True
    assert out["is_completed"] is True
    assert out["status_text"] == "Final"


def test_parse_flashscore_tournament_page_extracts_completed_match():
    html = """
    <script>
    cjs.initialFeeds["summary-results"] = {
        data: `SA?2?~ZA?ATP - SINGLES: Miami (USA), hard?~AA?lKA6R02T?AD?1774368600?ADE?1774368600?ER?1/8-finals?WN?FRI?PY?UH09iaZ6?AF?Fritz T.?WV?fritz-taylor?CB?200?BB?4?BD?7?BF?2?WM?LEH?PX?6PlgfXKR?AE?Lehecka J.?WU?lehecka-jiri?CA?62?BA?6?BC?6?BE?6?`
    };
    </script>
    """
    out = _parse_flashscore_tournament_page(html, default_tournament="ATP Miami", draw="ATP")
    assert len(out) == 1
    assert out[0]["players"][0]["name"] == "Lehecka J."
    assert out[0]["players"][0]["score_tokens"] == ["6", "6", "6"]
    assert out[0]["players"][1]["score_tokens"] == ["4", "7", "2"]
    assert out[0]["is_completed"] is True


def test_parse_flashscore_tournament_page_extracts_challenger_final():
    html = """
    <script>
    cjs.initialFeeds["summary-results"] = {
        data: `SA?2?~ZA?Challenger Men - Singles: Morelia (Mexico), hard?~AA?vLslGYyH?AD?1774372200?ADE?1774372200?ER?1/16-finals?WM?KOZ?PX?hI2zuuMo?AE?Kozlov S.?WU?kozlov-stefan?CA?200?BA?2?BC?2?WN?MAR?PY?h2AG5rdm?AF?Martin A.?WV?martin-andres?CB?200?BB?6?BD?6?`
    };
    </script>
    """
    out = _parse_flashscore_tournament_page(html, default_tournament="Morelia Challenger Men", draw="ATP")
    assert len(out) == 1
    assert out[0]["players"][0]["name"] == "Kozlov S."
    assert out[0]["players"][1]["name"] == "Martin A."
    assert out[0]["players"][0]["score_tokens"] == ["2", "2"]
    assert out[0]["players"][1]["score_tokens"] == ["6", "6"]
