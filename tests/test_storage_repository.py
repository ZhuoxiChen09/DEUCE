import asyncio

from deuce.providers.mock import MockLiveScoreProvider
from deuce.storage.database import init_db, make_engine, make_session_factory
from deuce.storage.repository import MatchRepository


def test_upsert_and_list_match():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    sf = make_session_factory(engine)
    repo = MatchRepository()
    row = asyncio.run(MockLiveScoreProvider().fetch_matches([]))[0]

    s = sf()
    try:
        m = repo.upsert_match_from_provider_row(s, row)
        s.commit()
        mid = m.id
        assert m.match_key
        m2 = repo.upsert_match_from_provider_row(s, row)
        s.commit()
        assert m2.id == mid
        listed = repo.list_matches_for_date(s, row["date"])
        assert len(listed) == 1
    finally:
        s.close()
