"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from deuce.api.routes import health, imports, ingest, matches, predictions, providers, simulation
from deuce.config.settings import import_enabled, live_scores_enabled, sqlite_auto_create
from deuce.jobs.scheduler import build_import_scheduler, scheduled_import
from deuce.providers.disabled import DisabledLiveScoreProvider
from deuce.providers.health import InstrumentedLiveScoreProvider, ProviderHealthTracker
from deuce.providers.interface import LiveScoreProvider, default_live_score_provider
from deuce.services.orchestrator import MatchOrchestrator
from deuce.services.prediction_service import LivePredictionService
from deuce.storage.database import init_db, make_engine, make_session_factory


def create_app(
    *,
    db_url: str | None = None,
    live_provider: LiveScoreProvider | None = None,
    import_adapter: object | None = None,
    prediction_service: LivePredictionService | None = None,
    start_scheduler: bool | None = None,
) -> FastAPI:
    engine = make_engine(db_url)
    url = str(engine.url)
    if sqlite_auto_create(url):
        init_db(engine)
    session_factory = make_session_factory(engine)
    health_tracker = ProviderHealthTracker()
    if live_provider is not None:
        base_provider = live_provider
    elif live_scores_enabled():
        base_provider = default_live_score_provider()
    else:
        base_provider = DisabledLiveScoreProvider()
    wrapped = InstrumentedLiveScoreProvider(base_provider, health_tracker)
    service = prediction_service or LivePredictionService()
    schedule = import_enabled() if start_scheduler is None else bool(start_scheduler)

    application = FastAPI(title="DEUCE", version="0.2.0", lifespan=_lifespan(engine))
    application.state.engine = engine
    application.state.session_factory = session_factory
    application.state.health_tracker = health_tracker
    application.state.live_provider = wrapped
    application.state.live_scores_enabled = not isinstance(base_provider, DisabledLiveScoreProvider)
    application.state.prediction_service = service
    application.state.model_loaded = service.model_ready
    application.state.import_adapter = import_adapter
    application.state.orchestrator = MatchOrchestrator(
        session_factory=session_factory,
        provider=wrapped,
    )
    application.state.scheduler = None
    if schedule:
        application.state.scheduler = build_import_scheduler(
            lambda: scheduled_import(session_factory, application.state.import_adapter)
        )

    application.include_router(health.router)
    application.include_router(matches.router)
    application.include_router(predictions.router)
    application.include_router(providers.router)
    application.include_router(simulation.router)
    application.include_router(ingest.router)
    application.include_router(imports.router)
    return application


def _lifespan(engine):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler = app.state.scheduler
        if scheduler is not None:
            scheduler.start()
        try:
            yield
        finally:
            if scheduler is not None and getattr(scheduler, "running", False):
                scheduler.shutdown(wait=False)
            engine.dispose()

    return lifespan


app = create_app()
