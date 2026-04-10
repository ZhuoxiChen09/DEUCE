import pytest

from deuce.backtest.engine import BacktestConfig, run_backtest, summarize_backtest


def test_run_backtest_simple():
    rows = [
        {
            "model_prob_a": 0.65,
            "outcome_a_won": 1,
            "market_price_a": 0.55,
            "market_price_b": 0.48,
            "tour": "ATP",
            "surface": "Hard",
            "tournament_tier": "M",
            "favorite_side": "A",
            "coverage_bucket": "high",
        },
        {
            "model_prob_a": 0.4,
            "outcome_a_won": 0,
            "market_price_a": 0.45,
            "market_price_b": 0.52,
            "tour": "ATP",
            "surface": "Clay",
            "tournament_tier": "500",
            "favorite_side": "B",
            "coverage_bucket": "low",
        },
    ]
    r = run_backtest(rows, config=BacktestConfig(edge_threshold=0.02, stake_mode="flat"))
    assert r.n_bets == 2
    assert len(r.bankroll_series) == 3
    assert len(r.trades) == 2
    assert r.total_staked > 0


def test_no_bet_filtering_by_edge_threshold():
    rows = [{"model_prob_a": 0.51, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50}]
    r = run_backtest(rows, config=BacktestConfig(edge_threshold=0.03, stake_mode="flat", flat_stake=10.0))
    assert r.n_bets == 0
    assert r.n_skipped == 1


def test_side_selection_correctness_b_side():
    rows = [{"model_prob_a": 0.40, "outcome_a_won": 0, "market_price_a": 0.52, "market_price_b": 0.48}]
    r = run_backtest(rows, config=BacktestConfig(edge_threshold=0.02, stake_mode="flat", flat_stake=10.0))
    assert r.n_bets == 1
    assert r.trades[0].side == "B"


def test_bankroll_evolution_win_then_loss():
    rows = [
        {"model_prob_a": 0.65, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50},
        {"model_prob_a": 0.65, "outcome_a_won": 0, "market_price_a": 0.50, "market_price_b": 0.50},
    ]
    cfg = BacktestConfig(initial_bankroll=1000.0, edge_threshold=0.03, stake_mode="flat", flat_stake=100.0)
    r = run_backtest(rows, config=cfg)
    # First trade win: +100, second loss: -100, net zero.
    assert r.total_pnl == pytest.approx(0.0)
    assert r.bankroll_series[-1] == pytest.approx(1000.0)
    assert r.roi == pytest.approx(0.0)


def test_roi_computation():
    rows = [{"model_prob_a": 0.70, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50}]
    cfg = BacktestConfig(initial_bankroll=1000.0, edge_threshold=0.03, stake_mode="flat", flat_stake=100.0)
    r = run_backtest(rows, config=cfg)
    assert r.total_staked == pytest.approx(100.0)
    assert r.total_pnl == pytest.approx(100.0)
    assert r.roi == pytest.approx(1.0)
    assert r.bankroll_return == pytest.approx(0.1)


def test_kelly_stake_sizing_and_bounds():
    rows = [{"model_prob_a": 0.70, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50}]
    cfg = BacktestConfig(
        initial_bankroll=1000.0,
        edge_threshold=0.01,
        stake_mode="kelly",
        kelly_fraction=0.25,
        kelly_max_bankroll_fraction=0.05,
        min_stake=1.0,
        max_stake=40.0,
    )
    r = run_backtest(rows, config=cfg)
    assert r.n_bets == 1
    assert 1.0 <= r.trades[0].stake <= 40.0


def test_no_vig_prices_do_not_change_realized_pnl_or_kelly_stake():
    rows = [{"model_prob_a": 0.70, "outcome_a_won": 1, "market_price_a": 0.60, "market_price_b": 0.50}]

    fair_edge = run_backtest(
        rows,
        config=BacktestConfig(
            initial_bankroll=1000.0,
            edge_threshold=0.03,
            stake_mode="kelly",
            kelly_fraction=0.25,
            kelly_max_bankroll_fraction=1.0,
            use_no_vig=True,
        ),
    )
    raw_edge = run_backtest(
        rows,
        config=BacktestConfig(
            initial_bankroll=1000.0,
            edge_threshold=0.03,
            stake_mode="kelly",
            kelly_fraction=0.25,
            kelly_max_bankroll_fraction=1.0,
            use_no_vig=False,
        ),
    )

    assert fair_edge.n_bets == 1
    assert raw_edge.n_bets == 1
    assert fair_edge.trades[0].stake == pytest.approx(raw_edge.trades[0].stake)
    assert fair_edge.total_pnl == pytest.approx(raw_edge.total_pnl)
    assert fair_edge.trades[0].market_prob == pytest.approx(0.60)


def test_sharpe_on_deterministic_sequence():
    rows = [
        {"model_prob_a": 0.70, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50},
        {"model_prob_a": 0.70, "outcome_a_won": 0, "market_price_a": 0.50, "market_price_b": 0.50},
        {"model_prob_a": 0.70, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50},
        {"model_prob_a": 0.70, "outcome_a_won": 0, "market_price_a": 0.50, "market_price_b": 0.50},
    ]
    cfg = BacktestConfig(initial_bankroll=1000.0, edge_threshold=0.01, stake_mode="flat", flat_stake=100.0)
    r = run_backtest(rows, config=cfg)
    assert r.n_bets == 4
    # Returns alternate +1,-1,+1,-1, mean zero -> Sharpe 0.
    assert r.sharpe_ratio == pytest.approx(0.0)


def test_summarize_backtest_api():
    rows = [{"model_prob_a": 0.70, "outcome_a_won": 1, "market_price_a": 0.50, "market_price_b": 0.50}]
    r = run_backtest(rows, config=BacktestConfig(stake_mode="flat", flat_stake=100.0, edge_threshold=0.01))
    s = summarize_backtest(r)
    assert s["n_bets"] == 1.0
    assert "roi" in s and "sharpe_ratio" in s and "log_loss" in s


def test_legacy_evaluate_predictions():
    from backtest import evaluate_predictions

    pred = [{"model_prob_a": 0.7}, {"model_prob_a": 0.3}]
    out = [1, 0]
    m = evaluate_predictions(pred, out)
    assert "log_loss" in m
    assert "brier_score" in m
