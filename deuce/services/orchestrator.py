"""Single-tick ingestion: fetch → dedup → persist matches, events, snapshots."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from deuce.pipeline.dedup import EventDeduplicator
from deuce.pipeline.events import EventType
from deuce.providers.interface import LiveScoreProvider
from deuce.providers.normalization import match_config_from_provider_dict
from deuce.storage.repository import MatchRepository


class MatchOrchestrator:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        repository: MatchRepository | None = None,
        deduplicator: EventDeduplicator | None = None,
        provider: LiveScoreProvider,
    ) -> None:
        self._session_factory = session_factory
        self._repo = repository or MatchRepository()
        self._dedup = deduplicator or EventDeduplicator()
        self._provider = provider

    @property
    def deduplicator(self) -> EventDeduplicator:
        return self._dedup

    async def ingest_dates(self, date_strs: list[str]) -> dict[str, Any]:
        rows = await self._provider.fetch_matches(date_strs)
        processed = 0
        skipped_dup = 0
        errors: list[str] = []

        session = self._session_factory()
        try:
            for row in rows:
                cfg = match_config_from_provider_dict(row)
                if cfg is None:
                    errors.append("skip: bad row")
                    continue
                key = cfg.match_id
                if self._dedup.is_duplicate(key, row):
                    skipped_dup += 1
                    continue
                try:
                    m = self._repo.upsert_match_from_provider_row(session, row)
                    seq = self._repo.next_event_sequence(session, m)
                    self._repo.append_event(
                        session,
                        m,
                        seq,
                        EventType.PROVIDER_ROW.value,
                        row,
                    )
                    self._repo.add_snapshot(session, m, self._provider.provider_id, row)
                    processed += 1
                except Exception as exc:
                    errors.append(str(exc))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        return {
            "fetched": len(rows),
            "processed": processed,
            "skipped_duplicate": skipped_dup,
            "errors": errors,
            "provider_id": self._provider.provider_id,
        }
