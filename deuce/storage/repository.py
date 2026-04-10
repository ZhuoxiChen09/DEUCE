"""CRUD helpers for live matches, events, snapshots, and predictions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from deuce.domain.models import MatchState
from deuce.providers.normalization import match_config_from_provider_dict, match_state_from_provider_dict
from deuce.storage.schemas import EventRow, MatchRow, PredictionRow, RawPayloadRow, SnapshotRow


def _row_to_json(row: dict[str, Any]) -> str:
    return json.dumps(row, default=str, separators=(",", ":"))


def _state_to_json(state: MatchState) -> str:
    return state.model_dump_json()


class MatchRepository:
    def upsert_match_from_provider_row(self, session: Session, row: dict[str, Any]) -> MatchRow:
        cfg = match_config_from_provider_dict(row)
        if cfg is None:
            raise ValueError("Cannot derive match_key from provider row")
        key = cfg.match_id
        state = match_state_from_provider_dict(row)
        domain_json = _state_to_json(state) if state is not None else None

        existing = session.scalar(select(MatchRow).where(MatchRow.match_key == key))
        raw_json = _row_to_json(row)
        common = dict(
            date_str=str(row.get("date") or ""),
            tournament=str(row.get("tournament") or ""),
            draw=str(row.get("draw") or ""),
            source=str(row.get("source") or ""),
            status_text=str(row.get("status_text") or ""),
            is_live=1 if row.get("is_live") else 0,
            is_completed=1 if row.get("is_completed") else 0,
            is_scheduled=1 if row.get("is_scheduled") else 0,
            raw_row_json=raw_json,
            domain_state_json=domain_json,
            updated_at=datetime.utcnow(),
        )
        if existing:
            for k, v in common.items():
                setattr(existing, k, v)
            session.flush()
            return existing
        m = MatchRow(match_key=key, **common)
        session.add(m)
        session.flush()
        return m

    def list_matches_for_date(self, session: Session, date_str: str) -> list[MatchRow]:
        stmt = select(MatchRow).where(MatchRow.date_str == date_str).order_by(MatchRow.tournament, MatchRow.id)
        return list(session.scalars(stmt).all())

    def get_by_match_key(self, session: Session, match_key: str) -> Optional[MatchRow]:
        return session.scalar(select(MatchRow).where(MatchRow.match_key == match_key))

    def next_event_sequence(self, session: Session, match: MatchRow) -> int:
        m = session.scalar(select(func.max(EventRow.sequence)).where(EventRow.match_id == match.id))
        return (m or 0) + 1

    def append_event(
        self,
        session: Session,
        match: MatchRow,
        sequence: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> EventRow:
        ev = EventRow(
            match_id=match.id,
            sequence=sequence,
            event_type=event_type,
            payload_json=_row_to_json(payload),
        )
        session.add(ev)
        session.flush()
        return ev

    def add_snapshot(
        self,
        session: Session,
        match: MatchRow,
        provider_id: str,
        payload: dict[str, Any],
    ) -> SnapshotRow:
        snap = SnapshotRow(
            match_id=match.id,
            provider_id=provider_id,
            payload_json=_row_to_json(payload),
        )
        session.add(snap)
        session.flush()
        return snap

    def add_prediction(
        self,
        session: Session,
        match: MatchRow,
        *,
        model_id: str,
        prob_a: float | None,
        prob_b: float | None,
        extras: dict[str, Any] | None = None,
    ) -> PredictionRow:
        pr = PredictionRow(
            match_id=match.id,
            model_id=model_id,
            prob_a=prob_a,
            prob_b=prob_b,
            extras_json=_row_to_json(extras or {}),
        )
        session.add(pr)
        session.flush()
        return pr

    def store_raw_payload(self, session: Session, source: str, payload: dict[str, Any]) -> RawPayloadRow:
        row = RawPayloadRow(source=source, payload_json=_row_to_json(payload))
        session.add(row)
        session.flush()
        return row
