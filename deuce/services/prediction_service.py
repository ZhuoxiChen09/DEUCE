"""Pre-match probabilities from the committed calibrated logistic regression."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Optional

from deuce.config.settings import catalog_path, model_path, player_db_path
from deuce.domain.enums import MatchFormat, Surface
from deuce.domain.models import MatchConfig, Player, Tournament
from deuce.features.base_features import PlayerStats

logger = logging.getLogger("deuce.prediction")

DEFAULT_PLAYER_A = "Carlos Alcaraz"
DEFAULT_PLAYER_B = "Jannik Sinner"
ALLOWED_SURFACES = ("Hard", "Clay", "Grass", "Carpet")
_SURFACE_ENUM = {
    "Hard": Surface.HARD,
    "Clay": Surface.CLAY,
    "Grass": Surface.GRASS,
    "Carpet": Surface.CARPET,
}
_SURFACE_CONTEXT = {
    "Hard": ("hard_outdoor", "hard_indoor"),
    "Clay": ("clay",),
    "Grass": ("grass",),
    "Carpet": ("carpet", "hard_indoor", "hard_outdoor"),
}


class PredictionInputError(ValueError):
    """Caller input that cannot be scored."""


class LivePredictionService:
    """Pre-match style probabilities. Live scores do not feed the model."""

    def __init__(
        self,
        *,
        model: Any = None,
        model_core: Any = None,
        model_enhanced: Any = None,
        feature_names: Optional[list[str]] = None,
        medians: Optional[dict[str, float]] = None,
        model_file: Any = None,
        player_db: Optional[list[dict[str, Any]]] = None,
        catalog: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._model = model
        self._model_core = model_core
        self._model_enhanced = model_enhanced
        self._feature_names = feature_names
        self._medians = medians
        self._model_file = model_file
        self._player_db = player_db
        self._catalog = catalog
        self._load_error: str | None = None
        if self._model is None and self._model_core is None and self._model_enhanced is None:
            self._load_default_model()

    @property
    def model_ready(self) -> bool:
        return self._model is not None or self._model_core is not None or self._model_enhanced is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _load_default_model(self) -> None:
        path = model_path() if self._model_file is None else self._model_file
        self._model_file = path
        try:
            from deuce.training.predict import load_model

            bundle, names, medians = load_model(str(path))
        except Exception as exc:
            self._load_error = f"Model bundle unavailable: {exc}"
            logger.warning("%s", self._load_error)
            return
        self._model_core = bundle
        self._feature_names = list(names)
        self._medians = dict(medians)

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
            feature_names=self._feature_names,
            medians=self._medians,
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
        """Score a configured matchup with the loaded model."""
        return self.predict_matchup(
            config.player_a.name,
            config.player_b.name,
            surface=_surface_label(config.tournament.surface),
            match_id=config.match_id,
            tournament_name=config.tournament.name or None,
        )

    def predict_matchup(
        self,
        player_a: str,
        player_b: str,
        *,
        surface: str = "Hard",
        match_id: str = "demo",
        tournament_name: str | None = None,
        stats_a: PlayerStats | None = None,
        stats_b: PlayerStats | None = None,
    ) -> dict[str, Any]:
        left, right, surface_label = _validate_matchup(player_a, player_b, surface)
        if not self.model_ready:
            return _unavailable(
                match_id,
                left,
                right,
                surface_label,
                self._load_error or "Model bundle is not loaded.",
            )
        catalog = self._catalog_rows()
        context_name = tournament_name or surface_context_name(surface_label, catalog)
        if stats_a is None or stats_b is None:
            from deuce.training.features import build_player_stats_from_datasets

            rows = self._player_rows()
            stats_a = stats_a or build_player_stats_from_datasets(left, player_db=rows)
            stats_b = stats_b or build_player_stats_from_datasets(right, player_db=rows)
        config = _match_config(match_id, left, right, surface_label, context_name)
        result = self.predict_with_stats(config, stats_a, stats_b, tournament_catalog=catalog)
        return _from_result(result, match_id=match_id, player_a=left, player_b=right, surface=surface_label)

    def predict_historical(self, row: Any) -> dict[str, Any]:
        payload = _payload_dict(getattr(row, "payload_json", "") or "")
        stats_a = _stats_from_stored(row.player_a, row.rank_a, row.points_a, payload, "a")
        stats_b = _stats_from_stored(row.player_b, row.rank_b, row.points_b, payload, "b")
        return self.predict_matchup(
            row.player_a,
            row.player_b,
            surface=row.surface or "Hard",
            match_id=row.match_key,
            tournament_name=row.tournament or None,
            stats_a=stats_a,
            stats_b=stats_b,
        )

    def _player_rows(self) -> list[dict[str, Any]]:
        if self._player_db is None:
            from deuce.player_db import load_db

            self._player_db = load_db(str(player_db_path()))
        return self._player_db

    def _catalog_rows(self) -> list[dict[str, Any]]:
        if self._catalog is None:
            path = catalog_path()
            if not path.exists():
                self._catalog = []
            else:
                from deuce.tournament_catalog import load_catalog

                self._catalog = load_catalog(str(path))
        return self._catalog

    @staticmethod
    def prediction_to_dict(result: Any) -> dict[str, Any]:
        return asdict(result)


def surface_context_name(surface: str, catalog: list[dict[str, Any]]) -> str:
    targets = _SURFACE_CONTEXT.get(surface, ())
    for entry in catalog:
        if entry.get("surface") in targets and entry.get("name"):
            return str(entry["name"])
    if surface == "Grass":
        return "Wimbledon"
    if surface == "Clay":
        return "Roland Garros"
    return "Australian Open"


def _validate_matchup(player_a: str, player_b: str, surface: str) -> tuple[str, str, str]:
    from deuce.name_utils import normalize_player_name

    left = (player_a or "").strip()
    right = (player_b or "").strip()
    if not left or not right:
        raise PredictionInputError("player_a and player_b are required")
    label = (surface or "").strip().title()
    if label not in ALLOWED_SURFACES:
        raise PredictionInputError("surface must be Hard, Clay, Grass, or Carpet")
    if normalize_player_name(left) == normalize_player_name(right):
        raise PredictionInputError("player_a and player_b must be different players")
    return left, right, label


def _surface_label(surface: Surface | str) -> str:
    if isinstance(surface, Surface):
        return surface.value.title()
    return str(surface or "Hard").strip().title()


def _match_config(
    match_id: str,
    player_a: str,
    player_b: str,
    surface: str,
    tournament_name: str,
) -> MatchConfig:
    return MatchConfig(
        match_id=match_id,
        tournament=Tournament(name=tournament_name, surface=_SURFACE_ENUM[surface]),
        player_a=Player(player_id="a", name=player_a),
        player_b=Player(player_id="b", name=player_b),
        match_format=MatchFormat.BEST_OF_3,
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats_from_stored(
    name: str,
    rank: Any,
    points: Any,
    payload: dict[str, Any],
    side: str,
) -> PlayerStats:
    def raw(key: str) -> Any:
        return payload.get(f"{key}_{side}")

    return PlayerStats(
        name=name,
        rank=_optional_float(rank),
        points=_optional_float(points),
        serve_rating=_optional_float(raw("serve_rating")),
        return_rating=_optional_float(raw("return_rating")),
        pressure_rating=_optional_float(raw("pressure_rating")),
        winloss_index=_optional_float(raw("winloss_index")),
        tdi_perf=_optional_float(raw("tdi_perf")),
    )


def _payload_dict(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _unavailable(match_id: str, player_a: str, player_b: str, surface: str, message: str) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "player_a": player_a,
        "player_b": player_b,
        "available": False,
        "message": message,
        "surface": surface,
        "model_prob_a": None,
        "model_prob_b": None,
        "raw_prob_a": None,
        "calibrated_prob_a": None,
        "model_used": None,
        "confidence": None,
        "data_completeness": None,
        "warnings": [],
    }


def _from_result(result: Any, *, match_id: str, player_a: str, player_b: str, surface: str) -> dict[str, Any]:
    used_model = result.confidence == "model" and result.model_used != "fallback"
    message = (
        "Calibrated logistic regression probability from the committed core bundle."
        if used_model
        else "Model prediction was not available; no calibrated probability was returned."
    )
    return {
        "match_id": match_id,
        "player_a": player_a,
        "player_b": player_b,
        "available": used_model,
        "message": message,
        "surface": surface,
        "model_prob_a": result.model_prob_a,
        "model_prob_b": result.model_prob_b,
        "raw_prob_a": result.raw_prob_a,
        "calibrated_prob_a": result.calibrated_prob_a,
        "model_used": result.model_used,
        "confidence": result.confidence,
        "data_completeness": result.data_completeness,
        "warnings": list(result.warnings or []),
    }
