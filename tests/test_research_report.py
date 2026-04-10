from deuce.research.report import research_report_from_rows


def test_research_report_basic():
    rows = [
        {"model_prob_a": 0.7, "outcome_a_won": 1, "tour": "ATP", "surface": "Hard"},
        {"model_prob_a": 0.4, "outcome_a_won": 0, "tour": "ATP", "surface": "Clay"},
    ]
    r = research_report_from_rows(rows)
    assert r.n == 2
    assert r.log_loss > 0
