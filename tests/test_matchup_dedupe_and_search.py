from cli import dedupe_matchup_rows
from deuce.presentation.kpi import filter_matchups


def test_dedupe_matchup_rows_keeps_single_canonical_matchup():
    rows = [
        {
            "event": "Novak Djokovic vs Carlos Alcaraz",
            "event_ticker": "EVT1",
            "when": "2026-03-23T20:00:00Z",
            "event_info_line": "",
            "raw_contract_tickers": ["A"],
            "canonical_key": ("EVT1", ("carlos alcaraz", "novak djokovic"), "2026-03-23"),
            "players": [{"name": "Novak Djokovic"}, {"name": "Carlos Alcaraz"}],
        },
        {
            "event": "Novak Djokovic vs Carlos Alcaraz",
            "event_ticker": "EVT1",
            "when": "2026-03-23T20:00:00Z",
            "event_info_line": "ATP Miami | Final",
            "event_round": "Final",
            "raw_contract_tickers": ["A", "B"],
            "canonical_key": ("EVT1", ("carlos alcaraz", "novak djokovic"), "2026-03-23"),
            "players": [{"name": "Novak Djokovic"}, {"name": "Carlos Alcaraz"}],
        },
    ]

    out = dedupe_matchup_rows(rows)
    assert len(out) == 1
    assert out[0]["event_info_line"] == "ATP Miami | Final"
    assert out[0]["raw_contract_tickers"] == ["A", "B"]


def test_filter_matchups_searches_live_score_and_event_context():
    matchups = [
        {
            "event": "Novak Djokovic vs Carlos Alcaraz",
            "event_info_line": "ATP Miami | Final",
            "event_competition": "ATP Miami",
            "event_round": "Final",
            "market_phase_label": "Live",
            "players": [
                {"name": "Novak Djokovic", "edge": 0.05, "yes_bid": 0.61},
                {"name": "Carlos Alcaraz", "edge": -0.05, "yes_bid": 0.39},
            ],
            "live_score": {
                "summary": "Final Set | Novak Djokovic: 6 4 3 | Carlos Alcaraz: 4 6 5",
                "status_text": "Final Set",
                "tournament": "Miami Open",
                "round": "Final",
                "players": [
                    {"name": "Novak Djokovic", "score_tokens": ["6", "4", "3"]},
                    {"name": "Carlos Alcaraz", "score_tokens": ["4", "6", "5"]},
                ],
            },
        }
    ]

    assert len(filter_matchups(matchups, search="miami")) == 1
    assert len(filter_matchups(matchups, search="final set")) == 1
    assert len(filter_matchups(matchups, search="6 4 3")) == 1
