"""Match events: types, payloads, and application to domain state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from deuce.domain.models import MatchConfig, MatchState
from deuce.providers.normalization import match_config_from_provider_dict, match_state_from_provider_dict


class EventType(str, Enum):
    PROVIDER_ROW = "provider_row"
    DOMAIN_SNAPSHOT = "domain_snapshot"


@dataclass(frozen=True)
class MatchEvent:
    match_key: str
    sequence: int
    event_type: EventType
    payload: dict[str, Any]


def apply_event(
    prior_state: MatchState | None,
    prior_config: MatchConfig | None,
    event: MatchEvent,
) -> tuple[Optional[MatchState], Optional[MatchConfig]]:
    """Return (new_state, config). Config is carried forward or rebuilt from provider rows."""
    if event.event_type == EventType.PROVIDER_ROW:
        row = event.payload
        cfg = match_config_from_provider_dict(row) or prior_config
        if cfg is None:
            return prior_state, prior_config
        state = match_state_from_provider_dict(row)
        return state, cfg
    if event.event_type == EventType.DOMAIN_SNAPSHOT:
        raw = event.payload.get("match_state")
        cfg_raw = event.payload.get("match_config")
        if raw is None:
            return prior_state, prior_config
        state = MatchState.model_validate(raw)
        cfg = MatchConfig.model_validate(cfg_raw) if cfg_raw else prior_config
        return state, cfg
    return prior_state, prior_config
