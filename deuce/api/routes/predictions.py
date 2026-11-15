from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from deuce.api.dependencies import get_prediction_service
from deuce.api.schemas import PredictionResponse
from deuce.services.prediction_service import (
    DEFAULT_PLAYER_A,
    DEFAULT_PLAYER_B,
    LivePredictionService,
    PredictionInputError,
)
from deuce.storage.schemas import HistoricalMatchRow

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/live/demo", response_model=PredictionResponse)
def live_prediction_demo(
    player_a: str = Query(DEFAULT_PLAYER_A),
    player_b: str = Query(DEFAULT_PLAYER_B),
    surface: str = Query("Hard"),
    svc: LivePredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Score a matchup with the committed calibrated logistic regression."""
    try:
        payload = svc.predict_matchup(player_a, player_b, surface=surface, match_id="demo")
    except PredictionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(**payload)


@router.get("/historical/{match_key}", response_model=PredictionResponse)
def historical_prediction(
    match_key: str,
    request: Request,
    svc: LivePredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Score an imported historical match from its stored pre-match fields."""
    session = request.app.state.session_factory()
    try:
        row = session.scalar(select(HistoricalMatchRow).where(HistoricalMatchRow.match_key == match_key))
    finally:
        session.close()
    if row is None:
        raise HTTPException(status_code=404, detail="historical match not found")
    try:
        payload = svc.predict_historical(row)
    except PredictionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictionResponse(**payload)
