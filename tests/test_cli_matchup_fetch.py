from cli import get_tennis_matchups


def test_get_tennis_matchups_fast_mode_disables_event_metadata(monkeypatch):
    captured = {}

    def _fake_build(ns):
        captured["fetch_event_title"] = ns.fetch_event_title
        captured["fetch_event_metadata"] = ns.fetch_event_metadata
        captured["max_results"] = ns.max_results
        return [], "ok"

    monkeypatch.setattr("cli._build_upcoming_matchups", _fake_build)

    rows, msg = get_tennis_matchups(player=None, days=3, max_results=80, fast=True)
    assert rows == []
    assert msg == "ok"
    assert captured["fetch_event_title"] is False
    assert captured["fetch_event_metadata"] is False
    assert captured["max_results"] == 80
