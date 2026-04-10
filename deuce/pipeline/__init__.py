"""Live ingestion pipeline: poll → dedup → reconcile → persist."""

from deuce.pipeline.dedup import EventDeduplicator, score_fingerprint
from deuce.pipeline.events import EventType, MatchEvent, apply_event
from deuce.pipeline.poller import LiveScorePoller
from deuce.pipeline.reconciliation import ScoreReconciler
from deuce.pipeline.retry import retry_async

__all__ = [
    "EventDeduplicator",
    "EventType",
    "LiveScorePoller",
    "MatchEvent",
    "ScoreReconciler",
    "apply_event",
    "retry_async",
    "score_fingerprint",
]
