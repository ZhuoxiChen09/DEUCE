"""FastAPI dependency hooks."""

from __future__ import annotations

from typing import Annotated

from fastapi import Request
from sqlalchemy.orm import Session, sessionmaker

from deuce.providers.health import ProviderHealthTracker
from deuce.providers.interface import LiveScoreProvider
from deuce.services.orchestrator import MatchOrchestrator
from deuce.services.prediction_service import LivePredictionService
from deuce.storage.repository import MatchRepository


def get_session_factory(request: Request) -> sessionmaker[Session]:
    return request.app.state.session_factory


def get_health_tracker(request: Request) -> ProviderHealthTracker:
    return request.app.state.health_tracker


def get_live_provider(request: Request) -> LiveScoreProvider:
    return request.app.state.live_provider


def get_repository() -> MatchRepository:
    return MatchRepository()


def get_prediction_service(request: Request) -> LivePredictionService:
    return request.app.state.prediction_service


def get_orchestrator(request: Request) -> MatchOrchestrator:
    return request.app.state.orchestrator


SessionFactoryDep = Annotated[sessionmaker[Session], get_session_factory]
TrackerDep = Annotated[ProviderHealthTracker, get_health_tracker]
ProviderDep = Annotated[LiveScoreProvider, get_live_provider]
