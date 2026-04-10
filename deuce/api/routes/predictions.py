from __future__ import annotations

from fastapi import APIRouter, Depends

from deuce.api.dependencies import get_prediction_service
from deuce.domain.models import MatchConfig, Player, Tournament
from deuce.domain.enums import MatchFormat, Surface
from deuce.api.schemas import PredictionPlaceholder
from deuce.services.prediction_service import LivePredictionService

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/live/demo", response_model=PredictionPlaceholder)
def live_prediction_demo(
    svc: LivePredictionService = Depends(get_prediction_service),
) -> PredictionPlaceholder:
    cfg = MatchConfig(
        match_id="demo",
        tournament=Tournament(name="Demo", surface=Surface.HARD),
        player_a=Player(player_id="a", name="Player A"),
        player_b=Player(player_id="b", name="Player B"),
        match_format=MatchFormat.BEST_OF_3,
    )
    d = svc.predict_placeholder(cfg)
    return PredictionPlaceholder(**d)
