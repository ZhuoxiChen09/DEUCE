"""Bet Candidates composite ranking."""

from deuce.presentation.kpi import board_rank_score, sort_matchups_for_bet_board


def _m(*, edge: float = 0.1, mq: str = "good", route: str = "core", comp: float = 0.8) -> dict:
    return {
        "players": [{"name": "A", "edge": edge}, {"name": "B", "edge": -edge}],
        "deuce": {"route": route, "data_completeness": comp},
        "market_quality": mq,
    }


def test_board_rank_demotes_thin_market():
    g = _m(mq="good")
    t = _m(mq="thin")
    assert board_rank_score(t) < board_rank_score(g)


def test_board_rank_demotes_legacy_route():
    c = _m(route="core")
    l = _m(route="legacy")
    assert board_rank_score(l) < board_rank_score(c)


def test_sort_tiebreaks_by_abs_edge():
    a = _m(edge=0.05)
    b = _m(edge=0.10)
    out = sort_matchups_for_bet_board([a, b])
    assert out[0]["players"][0]["edge"] == 0.10
