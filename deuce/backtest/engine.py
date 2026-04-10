"""Binary pre-match tennis backtest with explicit policy, sizing, and analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from deuce.bankroll.sizing import flat_stake, fractional_kelly_stake
from deuce.market.pricing import (
    MarketQuality,
    expected_value_per_dollar_risked,
    implied_prob_from_decimal_odds,
    no_vig_two_outcome,
)

StakeMode = Literal["flat", "fixed_fraction", "kelly"]


def _edge_bucket(max_edge: float) -> str:
    e = abs(max_edge)
    if e < 0.02:
        return "edge_<2pct"
    if e < 0.05:
        return "edge_2_5pct"
    if e < 0.10:
        return "edge_5_10pct"
    return "edge_>=10pct"


def _odds_bucket(price: float) -> str:
    """Coarse favorite/underdog band from fair price in (0,1)."""
    p = max(1e-6, min(1.0 - 1e-6, float(price)))
    if p >= 0.67:
        return "odds_heavy_fav"
    if p >= 0.55:
        return "odds_fav"
    if p >= 0.45:
        return "odds_coinflip"
    return "odds_dog"


@dataclass
class BacktestConfig:
    initial_bankroll: float = 1000.0
    edge_threshold: float = 0.03
    min_confidence: Optional[float] = None
    stake_mode: StakeMode = "flat"
    flat_stake: Optional[float] = None
    flat_fraction: float = 0.01  # backwards-compatible fallback when flat_stake is None
    fixed_bankroll_fraction: float = 0.02
    kelly_fraction: float = 0.25
    kelly_max_bankroll_fraction: float = 0.05
    min_stake: float = 0.0
    max_stake: Optional[float] = None
    max_stake_abs: Optional[float] = None
    min_model_completeness: float = 0.0
    min_market_quality: Optional[MarketQuality] = None
    use_no_vig: bool = True
    allow_bet_on_both_sides: bool = False
    skip_low_liquidity: bool = False


@dataclass
class BacktestResult:
    n_bets: int = 0
    n_skipped: int = 0
    wins: int = 0
    losses: int = 0
    total_staked: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    yield_per_bet: float = 0.0
    hit_rate: float = 0.0
    avg_edge: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_like: float = 0.0
    sharpe_ratio: float = 0.0
    bankroll_return: float = 0.0
    log_loss: float = 0.0
    brier: float = 0.0
    bankroll_series: List[float] = field(default_factory=list)
    pnl_series: List[float] = field(default_factory=list)
    trade_returns: List[float] = field(default_factory=list)
    decisions: List["BetDecision"] = field(default_factory=list)
    trades: List["TradeRecord"] = field(default_factory=list)
    segments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    segments_edge_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    segments_odds_bucket: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clv_placeholder_notes: str = "CLV: provide closing prices per bet to compute."


@dataclass(frozen=True)
class BetDecision:
    event_id: str
    matchup: str
    should_bet: bool
    side: Literal["A", "B", "none"]
    reason: str
    edge_a: float
    edge_b: float
    chosen_edge: float
    model_prob_a: float
    model_prob_b: float
    market_prob_a: float
    market_prob_b: float
    execution_prob_a: float
    execution_prob_b: float
    ev_per_dollar_a: float
    ev_per_dollar_b: float


@dataclass(frozen=True)
class TradeRecord:
    event_id: str
    matchup: str
    side: Literal["A", "B"]
    model_prob: float
    market_prob: float
    edge: float
    stake: float
    won: bool
    pnl: float
    bankroll_before: float
    bankroll_after: float


def _accumulate(seg: Dict[str, Dict[str, float]], key: str, pnl: float, stake: float) -> None:
    if key not in seg:
        seg[key] = {"n": 0.0, "pnl": 0.0, "staked": 0.0}
    seg[key]["n"] += 1.0
    seg[key]["pnl"] += pnl
    seg[key]["staked"] += stake


def _finalize_segment(seg: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k, v in seg.items():
        st = v["staked"]
        out[k] = {
            "n": v["n"],
            "pnl": v["pnl"],
            "roi": (v["pnl"] / st) if st > 0 else 0.0,
        }
    return out


def _segment_key(
    row: Dict[str, Any],
    *,
    tour: str = "tour",
    surface: str = "surface",
    tier: str = "tournament_tier",
    favorite: str = "favorite_side",
    coverage: str = "coverage_bucket",
) -> str:
    parts = [
        str(row.get(tour, "UNK")),
        str(row.get(surface, "UNK")),
        str(row.get(tier, "UNK")),
        str(row.get(favorite, "UNK")),
        str(row.get(coverage, "UNK")),
    ]
    return "|".join(parts)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _resolve_market_probs(
    row: Dict[str, Any],
    *,
    price_a_key: str,
    price_b_key: str,
    market_prob_a_key: str,
    market_prob_b_key: str,
    odds_a_key: str,
    odds_b_key: str,
    use_no_vig: bool,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve market probabilities from one of:
    - direct market probs
    - yes-price style (0..1)
    - decimal odds
    """
    mp_a = _safe_float(row.get(market_prob_a_key))
    mp_b = _safe_float(row.get(market_prob_b_key))
    if mp_a is not None:
        mp_a = max(0.0, min(1.0, mp_a))
    if mp_b is not None:
        mp_b = max(0.0, min(1.0, mp_b))

    if mp_a is None:
        mp_a = _safe_float(row.get(price_a_key))
    if mp_b is None:
        mp_b = _safe_float(row.get(price_b_key))

    if mp_a is None and (oa := _safe_float(row.get(odds_a_key))) is not None:
        mp_a = implied_prob_from_decimal_odds(oa)
    if mp_b is None and (ob := _safe_float(row.get(odds_b_key))) is not None:
        mp_b = implied_prob_from_decimal_odds(ob)

    if mp_a is None and mp_b is not None:
        mp_a = 1.0 - mp_b
    if mp_b is None and mp_a is not None:
        mp_b = 1.0 - mp_a
    if mp_a is None or mp_b is None:
        return None, None

    mp_a = max(0.0, min(1.0, mp_a))
    mp_b = max(0.0, min(1.0, mp_b))
    if use_no_vig:
        return no_vig_two_outcome(mp_a, mp_b)
    return mp_a, mp_b


def _resolve_execution_probs(
    row: Dict[str, Any],
    *,
    price_a_key: str,
    price_b_key: str,
    market_prob_a_key: str,
    market_prob_b_key: str,
    odds_a_key: str,
    odds_b_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve executable contract prices/probabilities.

    Unlike ``_resolve_market_probs()``, this never removes vig because realized PnL
    must be computed at the quoted execution price.
    """
    return _resolve_market_probs(
        row,
        price_a_key=price_a_key,
        price_b_key=price_b_key,
        market_prob_a_key=market_prob_a_key,
        market_prob_b_key=market_prob_b_key,
        odds_a_key=odds_a_key,
        odds_b_key=odds_b_key,
        use_no_vig=False,
    )


def _decide_bet(
    row: Dict[str, Any],
    *,
    cfg: BacktestConfig,
    model_prob_key: str,
    completeness_key: str,
    market_quality_key: str,
    confidence_key: str,
    price_a_key: str,
    price_b_key: str,
    market_prob_a_key: str,
    market_prob_b_key: str,
    odds_a_key: str,
    odds_b_key: str,
) -> BetDecision:
    p_a = _safe_float(row.get(model_prob_key))
    if p_a is None:
        return BetDecision("", "", False, "none", "missing model probability", 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0)
    p_a = max(0.0, min(1.0, p_a))
    p_b = 1.0 - p_a
    exec_a, exec_b = _resolve_execution_probs(
        row,
        price_a_key=price_a_key,
        price_b_key=price_b_key,
        market_prob_a_key=market_prob_a_key,
        market_prob_b_key=market_prob_b_key,
        odds_a_key=odds_a_key,
        odds_b_key=odds_b_key,
    )
    m_a, m_b = _resolve_market_probs(
        row,
        price_a_key=price_a_key,
        price_b_key=price_b_key,
        market_prob_a_key=market_prob_a_key,
        market_prob_b_key=market_prob_b_key,
        odds_a_key=odds_a_key,
        odds_b_key=odds_b_key,
        use_no_vig=cfg.use_no_vig,
    )
    if m_a is None or m_b is None or exec_a is None or exec_b is None:
        return BetDecision("", "", False, "none", "missing market probability/odds", 0.0, 0.0, 0.0, p_a, p_b, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0)

    edge_a = p_a - m_a
    edge_b = p_b - m_b
    ev_a = expected_value_per_dollar_risked(p_a, exec_a)
    ev_b = expected_value_per_dollar_risked(p_b, exec_b)

    comp = _safe_float(row.get(completeness_key))
    if comp is not None and comp < cfg.min_model_completeness:
        return BetDecision("", "", False, "none", "completeness below threshold", edge_a, edge_b, max(edge_a, edge_b), p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)
    conf = _safe_float(row.get(confidence_key))
    if cfg.min_confidence is not None and conf is not None and conf < cfg.min_confidence:
        return BetDecision("", "", False, "none", "confidence below threshold", edge_a, edge_b, max(edge_a, edge_b), p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)

    mq_raw = row.get(market_quality_key)
    mq = str(getattr(mq_raw, "value", mq_raw)).lower() if mq_raw is not None else ""
    if cfg.skip_low_liquidity and mq in ("thin", "one_sided"):
        return BetDecision("", "", False, "none", "low liquidity", edge_a, edge_b, max(edge_a, edge_b), p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)
    if cfg.min_market_quality is not None and mq:
        rank = {"ok": 0, "one_sided": 1, "thin": 2, "malformed": 3}
        if rank.get(mq, 99) > rank.get(cfg.min_market_quality.value, 0):
            return BetDecision("", "", False, "none", "market quality below threshold", edge_a, edge_b, max(edge_a, edge_b), p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)

    if not cfg.allow_bet_on_both_sides and edge_a >= cfg.edge_threshold and edge_b >= cfg.edge_threshold:
        if abs(edge_a - edge_b) < 1e-12:
            return BetDecision("", "", False, "none", "ambiguous both sides clear edge", edge_a, edge_b, max(edge_a, edge_b), p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)

    if edge_a >= cfg.edge_threshold and edge_a > edge_b:
        return BetDecision("", "", True, "A", "edge threshold passed", edge_a, edge_b, edge_a, p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)
    if edge_b >= cfg.edge_threshold and edge_b > edge_a:
        return BetDecision("", "", True, "B", "edge threshold passed", edge_a, edge_b, edge_b, p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)
    return BetDecision("", "", False, "none", "edge below threshold", edge_a, edge_b, max(edge_a, edge_b), p_a, p_b, m_a, m_b, exec_a, exec_b, ev_a, ev_b)


def _stake_for_decision(bankroll: float, decision: BetDecision, cfg: BacktestConfig) -> float:
    if cfg.stake_mode == "flat":
        stake = cfg.flat_stake if cfg.flat_stake is not None else flat_stake(bankroll, cfg.flat_fraction)
    elif cfg.stake_mode == "fixed_fraction":
        stake = max(0.0, bankroll * cfg.fixed_bankroll_fraction)
    else:
        market_prob = decision.execution_prob_a if decision.side == "A" else decision.execution_prob_b
        model_prob = decision.model_prob_a if decision.side == "A" else decision.model_prob_b
        stake = fractional_kelly_stake(
            bankroll,
            model_prob,
            market_prob,
            kelly_fraction=cfg.kelly_fraction,
            max_fraction_of_bankroll=cfg.kelly_max_bankroll_fraction,
        )
    stake = max(0.0, float(stake))
    hard_max = cfg.max_stake if cfg.max_stake is not None else cfg.max_stake_abs
    if hard_max is not None:
        stake = min(stake, float(hard_max))
    if cfg.min_stake > 0 and stake < cfg.min_stake:
        return 0.0
    return min(stake, bankroll)


def run_backtest(
    rows: Sequence[Dict[str, Any]],
    *,
    config: Optional[BacktestConfig] = None,
    outcome_key: str = "outcome_a_won",
    model_prob_key: str = "model_prob_a",
    market_prob_a_key: str = "market_prob_a",
    market_prob_b_key: str = "market_prob_b",
    price_a_key: str = "market_price_a",
    price_b_key: str = "market_price_b",
    odds_a_key: str = "decimal_odds_a",
    odds_b_key: str = "decimal_odds_b",
    completeness_key: str = "data_completeness",
    market_quality_key: str = "market_quality",
    confidence_key: str = "confidence_score_numeric",
    event_id_key: str = "event_id",
    matchup_key: str = "matchup",
    segment_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    bankroll = float(cfg.initial_bankroll)
    peak = bankroll
    max_dd = 0.0
    max_dd_pct = 0.0

    stakes: List[float] = []
    pnls: List[float] = []
    bankroll_path: List[float] = [bankroll]
    edges: List[float] = []
    trade_returns: List[float] = []
    decisions: List[BetDecision] = []
    trades: List[TradeRecord] = []
    probs: List[float] = []
    ys: List[int] = []

    seg_accum: Dict[str, Dict[str, float]] = {}
    seg_edge: Dict[str, Dict[str, float]] = {}
    seg_odds: Dict[str, Dict[str, float]] = {}

    n_bets = 0
    n_skip = 0
    wins = 0

    for row in rows:
        y_raw = row.get(outcome_key)
        p_raw = row.get(model_prob_key)
        p = _safe_float(p_raw)
        try:
            y = int(y_raw)
        except (TypeError, ValueError):
            n_skip += 1
            continue
        if p is not None:
            p = max(1e-15, min(1.0 - 1e-15, p))
            probs.append(p)
            ys.append(y)

        d = _decide_bet(
            row,
            cfg=cfg,
            model_prob_key=model_prob_key,
            completeness_key=completeness_key,
            market_quality_key=market_quality_key,
            confidence_key=confidence_key,
            price_a_key=price_a_key,
            price_b_key=price_b_key,
            market_prob_a_key=market_prob_a_key,
            market_prob_b_key=market_prob_b_key,
            odds_a_key=odds_a_key,
            odds_b_key=odds_b_key,
        )
        d = BetDecision(
            event_id=str(row.get(event_id_key, "")),
            matchup=str(row.get(matchup_key, row.get("event", ""))),
            should_bet=d.should_bet,
            side=d.side,
            reason=d.reason,
            edge_a=d.edge_a,
            edge_b=d.edge_b,
            chosen_edge=d.chosen_edge,
            model_prob_a=d.model_prob_a,
            model_prob_b=d.model_prob_b,
            market_prob_a=d.market_prob_a,
            market_prob_b=d.market_prob_b,
            execution_prob_a=d.execution_prob_a,
            execution_prob_b=d.execution_prob_b,
            ev_per_dollar_a=d.ev_per_dollar_a,
            ev_per_dollar_b=d.ev_per_dollar_b,
        )
        decisions.append(d)
        if not d.should_bet:
            n_skip += 1
            continue

        stake = _stake_for_decision(bankroll, d, cfg)
        if stake <= 0:
            n_skip += 1
            continue

        bet_a = d.side == "A"
        side_price = d.execution_prob_a if bet_a else d.execution_prob_b
        win_outcome = (y == 1) if bet_a else (y == 0)
        bankroll_before = bankroll
        contracts = stake / side_price if side_price > 0 else 0.0
        pnl = contracts * (1.0 if win_outcome else 0.0) - stake

        bankroll += pnl
        bankroll_after = bankroll
        bankroll_path.append(bankroll)
        stakes.append(stake)
        pnls.append(pnl)
        r = pnl / stake if stake > 0 else 0.0
        trade_returns.append(r)
        edges.append(abs(d.chosen_edge))
        n_bets += 1
        if win_outcome:
            wins += 1
        trades.append(
            TradeRecord(
                event_id=d.event_id,
                matchup=d.matchup,
                side=d.side,
                model_prob=d.model_prob_a if bet_a else d.model_prob_b,
                market_prob=side_price,
                edge=d.chosen_edge,
                stake=stake,
                won=win_outcome,
                pnl=pnl,
                bankroll_before=bankroll_before,
                bankroll_after=bankroll_after,
            )
        )

        if bankroll > peak:
            peak = bankroll
        dd = peak - bankroll
        if dd > max_dd:
            max_dd = dd
        if peak > 0:
            max_dd_pct = max(max_dd_pct, dd / peak)

        sk = segment_fn(row) if segment_fn else _segment_key(row)
        _accumulate(seg_accum, sk, pnl, stake)
        eb = _edge_bucket(d.chosen_edge)
        _accumulate(seg_edge, eb, pnl, stake)
        ob = _odds_bucket(side_price)
        _accumulate(seg_odds, ob, pnl, stake)

    total_staked = sum(stakes) if stakes else 0.0
    total_pnl = sum(pnls) if pnls else 0.0
    roi = (total_pnl / total_staked) if total_staked > 0 else 0.0
    yield_per_bet = (total_pnl / n_bets) if n_bets else 0.0
    hit_rate = (wins / n_bets) if n_bets else 0.0
    avg_edge = (sum(edges) / len(edges)) if edges else 0.0

    if len(trade_returns) >= 2:
        mean_r = sum(trade_returns) / len(trade_returns)
        var = sum((x - mean_r) ** 2 for x in trade_returns) / (len(trade_returns) - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        sharpe_like = mean_r / std
    else:
        sharpe_like = 0.0

    if probs and len(probs) == len(ys):
        ll = 0.0
        for prob, out in zip(probs, ys):
            p_win = prob if out == 1 else (1.0 - prob)
            ll -= math.log(max(1e-15, min(1.0 - 1e-15, p_win)))
        log_loss_val = ll / len(probs)
        brier = sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(probs)
    else:
        log_loss_val = 0.0
        brier = 0.0

    return BacktestResult(
        n_bets=n_bets,
        n_skipped=n_skip,
        wins=wins,
        losses=n_bets - wins,
        total_staked=total_staked,
        total_pnl=total_pnl,
        roi=roi,
        yield_per_bet=yield_per_bet,
        hit_rate=hit_rate,
        avg_edge=avg_edge,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        sharpe_like=sharpe_like,
        sharpe_ratio=sharpe_like,
        bankroll_return=(bankroll - cfg.initial_bankroll) / cfg.initial_bankroll if cfg.initial_bankroll > 0 else 0.0,
        log_loss=log_loss_val,
        brier=brier,
        bankroll_series=bankroll_path,
        pnl_series=pnls,
        trade_returns=trade_returns,
        decisions=decisions,
        trades=trades,
        segments=_finalize_segment(seg_accum),
        segments_edge_bucket=_finalize_segment(seg_edge),
        segments_odds_bucket=_finalize_segment(seg_odds),
    )


def summarize_backtest(result: BacktestResult) -> Dict[str, float]:
    """Concise scalar summary for reporting and UI."""
    return {
        "n_bets": float(result.n_bets),
        "hit_rate": float(result.hit_rate),
        "total_stake": float(result.total_staked),
        "total_pnl": float(result.total_pnl),
        "roi": float(result.roi),
        "bankroll_return": float(result.bankroll_return),
        "max_drawdown": float(result.max_drawdown),
        "max_drawdown_pct": float(result.max_drawdown_pct),
        "mean_return_per_bet": float(sum(result.trade_returns) / len(result.trade_returns)) if result.trade_returns else 0.0,
        "sharpe_ratio": float(result.sharpe_ratio),
        "log_loss": float(result.log_loss),
        "brier": float(result.brier),
    }


def summarize_legacy_predictions(
    predictions: Sequence[Dict[str, Any]],
    outcomes: Sequence[int],
) -> Dict[str, float]:
    """Backward-compatible wrapper: log loss + Brier only (from old backtest API)."""
    if not predictions or not outcomes or len(predictions) != len(outcomes):
        return {"log_loss": 0.0, "brier_score": 0.0, "roi": 0.0, "sharpe_ratio": 0.0}

    probs: List[float] = []
    for p in predictions:
        prob_a = p.get("model_prob_a") or p.get("prob_a")
        if prob_a is None:
            prob_a = 0.5
        probs.append(max(1e-15, min(1.0 - 1e-15, float(prob_a))))

    log_loss_val = 0.0
    for prob, out in zip(probs, outcomes):
        p_win = prob if out == 1 else (1.0 - prob)
        log_loss_val -= math.log(p_win)
    log_loss_val /= len(probs)
    brier = sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)

    return {
        "log_loss": round(log_loss_val, 4),
        "brier_score": round(brier, 4),
        "roi": 0.0,
        "sharpe_ratio": 0.0,
    }
