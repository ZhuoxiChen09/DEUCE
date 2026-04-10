from deuce.presentation.live_scoreboard_utils import split_score_tokens


def test_split_score_tokens_extracts_live_point_box():
    sets, live_point = split_score_tokens(["6", "4", "30"], is_live=True)
    assert sets == ["6", "4"]
    assert live_point == "30"


def test_split_score_tokens_keeps_set_scores_for_non_live_match():
    sets, live_point = split_score_tokens(["6", "4", "7"], is_live=False)
    assert sets == ["6", "4", "7"]
    assert live_point is None
