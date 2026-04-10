from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, sessionmaker

from deuce.api.dependencies import get_repository, get_session_factory
from deuce.api.schemas import MatchSummary
from deuce.storage.repository import MatchRepository

router = APIRouter(prefix="/matches", tags=["matches"])


@router.get("", response_model=list[MatchSummary])
def list_matches(
    date: Annotated[str, Query(..., description="Local date key YYYYMMDD", min_length=8, max_length=8)],
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
    repo: MatchRepository = Depends(get_repository),
) -> list[MatchSummary]:
    session = session_factory()
    try:
        rows = repo.list_matches_for_date(session, date)
        out: list[MatchSummary] = []
        for m in rows:
            out.append(
                MatchSummary(
                    match_key=m.match_key,
                    date_str=m.date_str,
                    tournament=m.tournament,
                    draw=m.draw,
                    source=m.source,
                    status_text=m.status_text,
                    is_live=bool(m.is_live),
                    is_completed=bool(m.is_completed),
                    is_scheduled=bool(m.is_scheduled),
                )
            )
        return out
    finally:
        session.close()


@router.get("/{match_key}/state")
def get_match_state(
    match_key: str,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
    repo: MatchRepository = Depends(get_repository),
) -> dict:
    session = session_factory()
    try:
        m = repo.get_by_match_key(session, match_key)
        if m is None:
            return {"found": False}
        raw = json.loads(m.domain_state_json) if m.domain_state_json else None
        return {"found": True, "match_key": match_key, "domain_state": raw}
    finally:
        session.close()
