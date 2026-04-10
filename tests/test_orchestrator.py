import pytest

from deuce.providers.mock import MockLiveScoreProvider
from deuce.services.orchestrator import MatchOrchestrator
from deuce.storage.database import init_db, make_engine, make_session_factory


@pytest.mark.asyncio
async def test_orchestrator_ingest_mock():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    sf = make_session_factory(engine)
    orch = MatchOrchestrator(session_factory=sf, provider=MockLiveScoreProvider())
    out = await orch.ingest_dates(["20260323"])
    assert out["fetched"] >= 1
    assert out["processed"] >= 1
    assert out["provider_id"] == "mock"
