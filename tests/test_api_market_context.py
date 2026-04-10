from datetime import datetime, timezone

from api import extract_tennis_round, format_iso8601_local, infer_market_start_state, summarize_tennis_event


def test_extract_tennis_round_from_title():
    title = "Will Aryna Sabalenka win the Sabalenka vs Zheng : Round Of 16 match?"
    assert extract_tennis_round(title) == "Round Of 16"


def test_infer_market_start_state_pregame():
    out = infer_market_start_state(
        "2026-03-23T21:20:00Z",
        now=datetime(2026, 3, 23, 20, 0, tzinfo=timezone.utc),
    )
    assert out["is_started"] is False
    assert out["phase"] == "pregame"
    assert out["scheduled_start_local"]


def test_infer_market_start_state_started():
    out = infer_market_start_state(
        "2026-03-23T21:20:00Z",
        now=datetime(2026, 3, 23, 22, 0, tzinfo=timezone.utc),
    )
    assert out["is_started"] is True
    assert out["phase"] == "started_or_live_like"


def test_summarize_tennis_event_extracts_competition_round_and_phase():
    payload = {
        "event": {
            "title": "Sabalenka vs Zheng",
            "sub_title": "Sabalenka vs Zheng (Mar 23)",
            "category": "Sports",
            "series_ticker": "KXWTAMATCH",
            "product_metadata": {
                "competition": "WTA Miami",
                "competition_scope": "Game",
            },
        },
        "markets": [
            {
                "title": "Will Aryna Sabalenka win the Sabalenka vs Zheng : Round Of 16 match?",
                "rules_primary": "If Aryna Sabalenka wins the Sabalenka vs Zheng professional tennis match in the 2026 WTA Miami Round Of 16 after a ball has been played, then the market resolves to Yes.",
                "expected_expiration_time": "2026-03-23T21:20:00Z",
            }
        ],
    }
    out = summarize_tennis_event(
        payload,
        now=datetime(2026, 3, 23, 20, 30, tzinfo=timezone.utc),
    )
    assert out["event_title"] == "Sabalenka vs Zheng"
    assert out["event_competition"] == "WTA Miami"
    assert out["event_round"] == "Round Of 16"
    assert out["event_info_line"] == "WTA Miami · Round Of 16"
    assert out["market_phase"] == "pregame"


def test_format_iso8601_local_returns_localized_display():
    out = format_iso8601_local("2026-03-23T21:20:00Z")
    assert out
