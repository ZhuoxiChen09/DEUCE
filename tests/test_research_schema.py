from deuce.presentation.research_schema import describe_oos_csv_gaps, missing_json_keys


def test_missing_json_keys_partial():
    data = {"mode": "wf", "log_loss": 0.5}
    miss = missing_json_keys(data)
    assert "brier" in miss
    assert "mode" not in miss


def test_describe_oos_csv_gaps():
    lines = describe_oos_csv_gaps({"foo"})
    assert any("cumulative_pnl" in x or "equity" in x for x in lines)
    assert any("model_prob_a" in x for x in lines)
