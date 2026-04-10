import asyncio
import json
from pathlib import Path

import pytest

from deuce.domain.enums import MatchStatus, ServerSide
from deuce.providers import (
    InstrumentedLiveScoreProvider,
    MockLiveScoreProvider,
    ProviderHealthTracker,
    ReplayLiveScoreProvider,
    SyncFnLiveScoreProvider,
    infer_server_side,
    match_state_from_provider_dict,
)


@pytest.mark.asyncio
async def test_mock_provider_returns_deterministic_rows():
    p = MockLiveScoreProvider()
    rows = await p.fetch_matches(["20260323"])
    assert rows[0]["tournament"] == "Mock Open"
    assert rows[0]["players"][0]["score_tokens"] == ["6", "0", "0"]


@pytest.mark.asyncio
async def test_sync_fn_provider_runs_in_thread():
    def sync_fetch(date_strs):
        return [{"date": d, "echo": True} for d in date_strs]

    p = SyncFnLiveScoreProvider("test", sync_fetch)
    out = await p.fetch_matches(["20260101"])
    assert out == [{"date": "20260101", "echo": True}]


@pytest.mark.asyncio
async def test_replay_provider_advances_snapshots():
    p = ReplayLiveScoreProvider(
        [
            [{"k": 1}],
            [{"k": 2}],
        ],
        repeat_last=False,
    )
    assert await p.fetch_matches([]) == [{"k": 1}]
    assert await p.fetch_matches([]) == [{"k": 2}]
    assert await p.fetch_matches([]) == []


@pytest.mark.asyncio
async def test_instrumented_provider_records_success():
    inner = MockLiveScoreProvider()
    tracker = ProviderHealthTracker()
    p = InstrumentedLiveScoreProvider(inner, tracker)
    await p.fetch_matches([])
    snap = tracker.snapshot(inner.provider_id)
    assert snap.success_count == 1
    assert snap.failure_count == 0
    assert snap.last_latency_ms is not None


@pytest.mark.asyncio
async def test_instrumented_provider_records_failure():
    class Boom:
        @property
        def provider_id(self):
            return "boom"

        async def fetch_matches(self, date_strs):
            raise RuntimeError("nope")

    tracker = ProviderHealthTracker()
    p = InstrumentedLiveScoreProvider(Boom(), tracker)
    with pytest.raises(RuntimeError):
        await p.fetch_matches([])
    snap = tracker.snapshot("boom")
    assert snap.failure_count == 1
    assert "nope" in (snap.last_error or "")


def test_infer_server_side_wta_and_atp():
    assert (
        infer_server_side({"serve": "1", "source": "wta-official-public"}) is ServerSide.PLAYER_A
    )
    assert (
        infer_server_side({"serve": "2", "source": "wta-official-public"}) is ServerSide.PLAYER_B
    )
    assert (
        infer_server_side({"serve": "0", "source": "atp-official-public"}) is ServerSide.PLAYER_A
    )
    assert (
        infer_server_side({"serve": "1", "source": "atp-official-public"}) is ServerSide.PLAYER_B
    )


def test_match_state_from_provider_live_row():
    row = {
        "date": "20260323",
        "tournament": "Miami Open",
        "draw": "WTA",
        "is_live": True,
        "is_completed": False,
        "is_scheduled": False,
        "serve": "2",
        "source": "wta-official-public",
        "players": [
            {"name": "A", "score_tokens": ["6", "0", "0"]},
            {"name": "B", "score_tokens": ["2", "1", "15"]},
        ],
    }
    state = match_state_from_provider_dict(row)
    assert state is not None
    assert state.status == MatchStatus.LIVE
    assert state.current_set is not None
    assert state.current_set.games_a == 0
    assert state.current_set.games_b == 1
    assert state.current_game is not None
    # B serves -> server frame (15, 0)
    assert state.current_game.point_state.server_points == "15"
    assert state.current_game.point_state.returner_points == "0"


def test_match_state_from_provider_completed_row():
    row = {
        "date": "20260323",
        "tournament": "Miami Open",
        "draw": "WTA",
        "is_live": False,
        "is_completed": True,
        "is_scheduled": False,
        "players": [
            {"name": "A", "score_tokens": ["6", "6"]},
            {"name": "B", "score_tokens": ["4", "3"]},
        ],
        "source": "espn",
    }
    state = match_state_from_provider_dict(row)
    assert state is not None
    assert state.status == MatchStatus.COMPLETED


def test_replay_from_json_file(tmp_path: Path):
    payload = {"snapshots": [[{"x": 1}], [{"x": 2}]]}
    path = tmp_path / "rep.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    p = ReplayLiveScoreProvider.from_json_file(path)
    assert asyncio.run(p.fetch_matches([])) == [{"x": 1}]
    assert asyncio.run(p.fetch_matches([])) == [{"x": 2}]
