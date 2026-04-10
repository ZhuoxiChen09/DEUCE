from deuce.domain.enums import MatchStatus
from deuce.pipeline.dedup import EventDeduplicator, score_fingerprint
from deuce.pipeline.events import EventType, MatchEvent, apply_event


def test_score_fingerprint_stable():
    row = {
        "status_text": "1st Set",
        "source": "mock",
        "players": [{"name": "A", "score_tokens": ["6", "0"]}, {"name": "B", "score_tokens": ["4", "1"]}],
    }
    assert score_fingerprint(row) == score_fingerprint(row)


def test_deduplicator_detects_repeat():
    d = EventDeduplicator()
    row = {"source": "s", "status_text": "x", "players": [{"name": "A", "score_tokens": ["1"]}, {"name": "B", "score_tokens": ["0"]}]}
    assert d.is_duplicate("k1", row) is False
    assert d.is_duplicate("k1", row) is True


def test_apply_event_provider_row():
    row = {
        "date": "20260323",
        "tournament": "T",
        "draw": "WTA",
        "is_live": True,
        "is_completed": False,
        "is_scheduled": False,
        "serve": "1",
        "source": "t",
        "players": [
            {"name": "A", "score_tokens": ["0", "15"]},
            {"name": "B", "score_tokens": ["0", "0"]},
        ],
    }
    ev = MatchEvent(match_key="x", sequence=1, event_type=EventType.PROVIDER_ROW, payload=row)
    state, cfg = apply_event(None, None, ev)
    assert cfg is not None
    assert state is not None
    assert state.status == MatchStatus.LIVE
