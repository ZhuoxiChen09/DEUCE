from __future__ import annotations

from fastapi import APIRouter

from deuce.api.schemas import SimulationRequest, SimulationResponse
from deuce.pipeline.reconciliation import ScoreReconciler

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/reconcile", response_model=SimulationResponse)
def reconcile_rows(body: SimulationRequest) -> SimulationResponse:
    recon = ScoreReconciler()
    states: list[dict] = []
    errors: list[str] = []
    for i, row in enumerate(body.rows):
        try:
            st = recon.reconcile(row)
            if st is None:
                errors.append(f"row {i}: no state")
                states.append({})
            else:
                states.append(st.model_dump(mode="json"))
        except Exception as exc:
            errors.append(f"row {i}: {exc}")
            states.append({})
    return SimulationResponse(states=states, errors=errors)
