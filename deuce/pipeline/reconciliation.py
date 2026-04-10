"""Coarse provider rows → domain MatchState (uses providers.normalization)."""

from __future__ import annotations

from typing import Any, Optional

from deuce.domain.models import MatchState
from deuce.providers.normalization import match_state_from_provider_dict


class ScoreReconciler:
    """Thin façade so pipeline code does not import normalization everywhere."""

    @staticmethod
    def reconcile(row: dict[str, Any]) -> Optional[MatchState]:
        return match_state_from_provider_dict(row)
