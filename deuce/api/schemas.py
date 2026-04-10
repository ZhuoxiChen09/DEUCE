"""Pydantic schemas for REST responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ProviderHealthOut(BaseModel):
    provider_id: str
    success_count: int = 0
    failure_count: int = 0
    last_latency_ms: Optional[float] = None
    last_error: Optional[str] = None
    last_success_ts: Optional[float] = None
    last_failure_ts: Optional[float] = None


class IngestRequest(BaseModel):
    dates: list[str] = Field(..., min_length=1, description="Local date keys YYYYMMDD")


class IngestResponse(BaseModel):
    fetched: int
    processed: int
    skipped_duplicate: int
    errors: list[str] = Field(default_factory=list)
    provider_id: str


class MatchSummary(BaseModel):
    match_key: str
    date_str: str
    tournament: str
    draw: str
    source: str
    status_text: str
    is_live: bool
    is_completed: bool
    is_scheduled: bool


class SimulationRequest(BaseModel):
    rows: list[dict[str, Any]]


class SimulationResponse(BaseModel):
    states: list[dict[str, Any]]
    errors: list[str] = Field(default_factory=list)


class PredictionPlaceholder(BaseModel):
    match_id: str
    player_a: str
    player_b: str
    available: bool = False
    message: str
