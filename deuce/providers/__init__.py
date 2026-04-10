"""Live score provider abstraction (Phase 2)."""

from deuce.providers.health import (
    InstrumentedLiveScoreProvider,
    ProviderHealth,
    ProviderHealthTracker,
)
from deuce.providers.interface import (
    LiveScoreProvider,
    SyncFnLiveScoreProvider,
    default_live_score_provider,
)
from deuce.providers.mock import MockLiveScoreProvider
from deuce.providers.normalization import (
    infer_server_side,
    match_config_from_provider_dict,
    match_state_from_provider_dict,
)
from deuce.providers.replay import ReplayLiveScoreProvider

__all__ = [
    "InstrumentedLiveScoreProvider",
    "LiveScoreProvider",
    "MockLiveScoreProvider",
    "ProviderHealth",
    "ProviderHealthTracker",
    "ReplayLiveScoreProvider",
    "SyncFnLiveScoreProvider",
    "default_live_score_provider",
    "infer_server_side",
    "match_config_from_provider_dict",
    "match_state_from_provider_dict",
]
