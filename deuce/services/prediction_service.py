"""Bridge to training `predict_match_probability` when stats and models are available."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Optional

from deuce.domain.models import MatchConfig


class LivePredictionService:
    """Pre-match style probabilities; live scores do not feed the model (leakage-safe)."""

    def __init__(
        self,
        *,
        model: Any = None,
        model_core: Any = None,
        model_enhanced: Any = None,
    ) -> None:
        self._model = model
        self._model_core = model_core
        self._model_enhanced = model_enhanced

    def predict_with_stats(
        self,
        config: MatchConfig,
        player_a_stats: Any,
        player_b_stats: Any,
        *,
        tournament_catalog: Any = None,
        match_context: Optional[dict[str, Any]] = None,
    ) -> Any:
        from deuce.training.predict import predict_match_probability

        return predict_match_probability(
            player_a_stats,
            player_b_stats,
            model=self._model,
            model_core=self._model_core,
            model_enhanced=self._model_enhanced,
            tournament_name=config.tournament.name or None,
            catalog=tournament_catalog,
            match_context=match_context,
        )

    async def predict_with_stats_async(
        self,
        config: MatchConfig,
        player_a_stats: Any,
        player_b_stats: Any,
        **kwargs: Any,
    ) -> Any:
        return await asyncio.to_thread(
            self.predict_with_stats,
            config,
            player_a_stats,
            player_b_stats,
            **kwargs,
        )

    def predict_placeholder(self, config: MatchConfig) -> dict[str, Any]:
        """Used when the API has names only (no leaderboard / DB stats wired)."""
        return {
            "match_id": config.match_id,
            "player_a": config.player_a.name,
            "player_b": config.player_b.name,
            "available": False,
            "message": "Provide player_a_stats and player_b_stats (PlayerStats) to run the model.",
        }

    @staticmethod
    def prediction_to_dict(result: Any) -> dict[str, Any]:
        return asdict(result)
