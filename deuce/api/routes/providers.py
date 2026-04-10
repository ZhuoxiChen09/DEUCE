from __future__ import annotations

from fastapi import APIRouter, Depends

from deuce.api.dependencies import get_health_tracker
from deuce.api.schemas import ProviderHealthOut
from deuce.providers.health import ProviderHealthTracker

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("/health", response_model=list[ProviderHealthOut])
def providers_health(
    tracker: ProviderHealthTracker = Depends(get_health_tracker),
) -> list[ProviderHealthOut]:
    snaps = tracker.all_snapshots()
    if not snaps:
        return []
    return [ProviderHealthOut(**s.__dict__) for s in snaps]
