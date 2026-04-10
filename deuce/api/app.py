"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from deuce.api.routes import health, ingest, matches, predictions, providers, simulation
from deuce.providers.health import InstrumentedLiveScoreProvider, ProviderHealthTracker
from deuce.providers.interface import LiveScoreProvider, default_live_score_provider
from deuce.services.orchestrator import MatchOrchestrator
from deuce.services.prediction_service import LivePredictionService
from deuce.storage.database import init_db, make_engine, make_session_factory


def create_app(
    *,
    db_url: str | None = None,
    live_provider: LiveScoreProvider | None = None,
) -> FastAPI:
    engine = make_engine(db_url)
    init_db(engine)
    session_factory = make_session_factory(engine)
    health_tracker = ProviderHealthTracker()
    base_provider = live_provider or default_live_score_provider()
    wrapped = InstrumentedLiveScoreProvider(base_provider, health_tracker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        engine.dispose()

    application = FastAPI(title="DEUCE", version="0.1.0", lifespan=lifespan)
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.health_tracker = health_tracker
    application.state.live_provider = wrapped
    application.state.prediction_service = LivePredictionService()
    application.state.orchestrator = MatchOrchestrator(
        session_factory=session_factory,
        provider=wrapped,
    )

    application.include_router(health.router)
    application.include_router(matches.router)
    application.include_router(predictions.router)
    application.include_router(providers.router)
    application.include_router(simulation.router)
    application.include_router(ingest.router)
    return application


app = create_app()
