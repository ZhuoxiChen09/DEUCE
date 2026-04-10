from deuce.bankroll.sizing import flat_stake, fractional_kelly_stake


def test_flat():
    assert flat_stake(1000, 0.01) == 10.0


def test_kelly_zero_when_no_edge():
    assert fractional_kelly_stake(1000, 0.5, 0.55) == 0.0
