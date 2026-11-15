from fastapi import APIRouter, Request

from deuce.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        live_scores_enabled=bool(getattr(request.app.state, "live_scores_enabled", False)),
        model_loaded=bool(getattr(request.app.state, "model_loaded", False)),
    )
