from __future__ import annotations

from fastapi import APIRouter, Depends

from deuce.api.dependencies import get_orchestrator
from deuce.api.schemas import IngestRequest, IngestResponse
from deuce.services.orchestrator import MatchOrchestrator

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_dates(
    body: IngestRequest,
    orch: MatchOrchestrator = Depends(get_orchestrator),
) -> IngestResponse:
    result = await orch.ingest_dates(body.dates)
    return IngestResponse(**result)
