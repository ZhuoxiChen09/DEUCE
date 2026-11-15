from __future__ import annotations

from fastapi import APIRouter, Request

from deuce.api.schemas import ImportRunOut
from deuce.ingestion.job import run_import

router = APIRouter(prefix="/imports", tags=["imports"])


@router.post("", response_model=ImportRunOut)
def run_import_now(request: Request) -> ImportRunOut:
    """Run the configured adapter once. Bad loads are blocked or flagged."""
    outcome = run_import(
        session_factory=request.app.state.session_factory,
        adapter=request.app.state.import_adapter,
    )
    return ImportRunOut(**outcome.to_dict())
