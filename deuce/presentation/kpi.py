"""Aggregate KPIs from enriched Kalshi matchup dicts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from deuce.presentation.bet_signal import decision_record, route_reliability_label
from deuce.presentation.formatting import UI_FAIL, UI_PASS, UI_WARN


def _matchup_search_blob(m: Dict[str, Any]) -> str:
    players = m.get("players") or []
    live_score = m.get("live_score") or {}
    live_players = live_score.get("players") or []
    parts = [
        " ".join(str(p.get("name") or "") for p in players),
        str(m.get("event") or m.get("title") or ""),
        str(m.get("event_info_line") or ""),
        str(m.get("event_competition") or ""),
        str(m.get("event_round") or ""),
        str(m.get("market_phase_label") or ""),
        str(live_score.get("summary") or ""),
        str(live_score.get("tournament") or ""),
        str(live_score.get("round") or ""),
        str(live_score.get("status_text") or ""),
        " ".join(str(p.get("name") or "") for p in live_players),
        " ".join(" ".join(p.get("score_tokens") or []) for p in live_players),
    ]
    return " ".join(part for part in parts if part).lower()


def summarize_matchups_kpis(matchups: List[Dict[str, Any]], *, show_predictions: bool) -> Dict[str, Any]:
    """Return counts and averages for terminal header."""
    n = len(matchups)
    out: Dict[str, Any] = {
        "n_markets": n,
        "n_priced": 0,
        "n_positive_edge": 0,
        "avg_edge": None,
        "max_edge": None,
        "avg_completeness": None,
        "pct_core": None,
        "pct_enhanced": None,
        "pct_legacy_or_fb": None,
    }
    if not show_predictions or not matchups:
        return out

    edges: List[float] = []
    comps: List[float] = []
    n_core = n_enh = n_other = 0

    for m in matchups:
        players = m.get("players") or []
        if len(players) != 2:
            continue
        deuce = m.get("deuce") or {}
        route = str(deuce.get("route") or m.get("prediction_source") or "")
        if route == "core":
            n_core += 1
        elif route == "enhanced":
            n_enh += 1
        else:
            n_other += 1

        c = deuce.get("data_completeness")
        if c is not None:
            comps.append(float(c))

        priced = False
        max_abs = 0.0
        for p in players:
            e = p.get("edge")
            if e is not None:
                priced = True
                ae = abs(float(e))
                edges.append(ae)
                max_abs = max(max_abs, ae)
        if priced:
            out["n_priced"] += 1
            if max_abs > 0.02:
                out["n_positive_edge"] += 1

    if edges:
        out["avg_edge"] = sum(edges) / len(edges)
        out["max_edge"] = max(edges)
    if comps:
        out["avg_completeness"] = sum(comps) / len(comps)
    tot_r = n_core + n_enh + n_other
    if tot_r > 0:
        out["pct_core"] = n_core / tot_r
        out["pct_enhanced"] = n_enh / tot_r
        out["pct_legacy_or_fb"] = n_other / tot_r
    return out


def filter_matchups(
    matchups: List[Dict[str, Any]],
    *,
    search: str = "",
    min_edge: float = 0.0,
    min_completeness: float = 0.0,
    route_filter: str = "all",
    bet_only: bool = False,
    min_edge_bet: float = 0.03,
    min_comp_bet: float = 0.55,
    sort_by: str = "edge",
    tour_filter: str = "all",
    market_quality_filter: str = "all",
) -> List[Dict[str, Any]]:
    """Filter and sort matchups (best-effort when fields missing)."""
    s = (search or "").strip().lower()
    rows: List[Dict[str, Any]] = []
    for m in matchups:
        players = m.get("players") or []
        if len(players) != 2:
            continue
        search_blob = _matchup_search_blob(m)
        ev_text = str(m.get("event") or m.get("title") or "").lower()
        if s and s not in search_blob:
            continue
        if tour_filter == "wta" and "wta" not in ev_text:
            continue
        if tour_filter == "atp" and "wta" in ev_text:
            continue
        mq_raw = str(m.get("market_quality") or "").strip().lower()
        if market_quality_filter == "exclude_thin" and mq_raw == "thin":
            continue
        if market_quality_filter == "good_only" and mq_raw in ("thin", "malformed"):
            continue
        deuce = m.get("deuce") or {}
        route = str(deuce.get("route") or m.get("prediction_source") or "")
        if route_filter != "all" and route != route_filter:
            continue
        comp = deuce.get("data_completeness")
        if comp is not None and float(comp) < min_completeness:
            continue
        e0 = players[0].get("edge")
        e1 = players[1].get("edge")
        ae = 0.0
        if e0 is not None:
            ae = max(ae, abs(float(e0)))
        if e1 is not None:
            ae = max(ae, abs(float(e1)))
        if ae < min_edge:
            continue
        if bet_only:
            er = bool(m.get("edge_reliable", False))
            li = bool(m.get("low_information_prediction", False))
            mv = bool(m.get("market_valid", True))
            ll = bool(m.get("low_liquidity", False))
            if not er or li or not mv or ll:
                continue
            if comp is not None and float(comp) < min_comp_bet:
                continue
            if ae < min_edge_bet:
                continue
        rows.append(m)

    def sort_key(mm: Dict[str, Any]) -> int:
        pl = mm.get("players") or []
        deuce = mm.get("deuce") or {}
        if sort_by == "completeness":
            c = deuce.get("data_completeness")
            return int(-(float(c) if c is not None else 0) * 1000)
        if sort_by == "confidence" and mm.get("evaluation"):
            ev = mm["evaluation"]
            mp = {"high": 3, "medium": 2, "low": 1}.get(getattr(ev, "confidence_score", "low"), 0)
            return -mp
        # edge default
        ae = 0.0
        for p in pl:
            e = p.get("edge")
            if e is not None:
                ae = max(ae, abs(float(e)))
        return int(-ae * 10000)

    rows.sort(key=sort_key)
    return rows


def classify_matchup_status(
    m: Dict[str, Any],
    *,
    min_edge: float,
    min_completeness: float,
) -> str:
    """Return decision_record.status for a single enriched matchup."""
    players = m.get("players") or []
    if len(players) != 2:
        return "no_bet"
    deuce = m.get("deuce") or {}
    route = str(deuce.get("route") or m.get("prediction_source") or "")
    e0 = players[0].get("edge")
    e1 = players[1].get("edge")
    ae = 0.0
    if e0 is not None:
        ae = max(ae, abs(float(e0)))
    if e1 is not None:
        ae = max(ae, abs(float(e1)))
    edge = float(e0) if e0 is not None and (e1 is None or abs(float(e0)) >= abs(float(e1))) else float(e1 or 0.0)
    comp = deuce.get("data_completeness")
    comp_f = float(comp) if comp is not None else None
    rec = decision_record(
        edge=edge,
        abs_edge=ae,
        completeness=comp_f,
        market_quality=str(m.get("market_quality") or ""),
        route=route,
        edge_reliable=bool(m.get("edge_reliable", False)),
        low_information=bool(m.get("low_information_prediction", False)),
        market_valid=bool(m.get("market_valid", True)),
        low_liquidity=bool(m.get("low_liquidity", False)),
        min_edge=min_edge,
        min_completeness=min_completeness,
    )
    return rec.status


def _max_abs_edge_matchup(m: Dict[str, Any]) -> float:
    ae = 0.0
    for p in m.get("players") or []:
        e = p.get("edge")
        if e is not None:
            ae = max(ae, abs(float(e)))
    return ae


def board_rank_score(m: Dict[str, Any]) -> float:
    """
    Higher = better priority on Bet Candidates board.

    Combines |edge| with route reliability, market quality, and completeness so thin markets
    do not always rank above cleaner setups for the same raw edge.
    """
    players = m.get("players") or []
    if len(players) != 2:
        return 0.0
    ae = _max_abs_edge_matchup(m)
    deuce = m.get("deuce") or {}
    comp = deuce.get("data_completeness")
    comp_f = float(comp) if comp is not None else 0.0
    route = str(deuce.get("route") or m.get("prediction_source") or "")
    rr = route_reliability_label(route)
    route_w = 1.0 if rr == "ok" else 0.88
    mq = str(m.get("market_quality") or "").strip().lower()
    if mq in ("thin", "malformed"):
        mq_w = 0.65
    elif mq == "":
        mq_w = 0.92
    else:
        mq_w = 1.0
    return ae * route_w * mq_w + 0.12 * comp_f


def _bet_board_sort_key(m: Dict[str, Any]) -> Tuple[float, float]:
    return (-board_rank_score(m), -_max_abs_edge_matchup(m))


def sort_matchups_for_bet_board(matchups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(matchups, key=_bet_board_sort_key)


def decision_counts_by_status(
    matchups: List[Dict[str, Any]],
    *,
    min_edge: float,
    min_completeness: float,
) -> Dict[str, int]:
    """Counts of bet_candidate / weak_edge / low_confidence / no_bet over 2-player matchups."""
    out: Dict[str, int] = {
        "bet_candidate": 0,
        "weak_edge": 0,
        "low_confidence": 0,
        "no_bet": 0,
    }
    for m in matchups:
        st_ = classify_matchup_status(m, min_edge=min_edge, min_completeness=min_completeness)
        if st_ in out:
            out[st_] += 1
    return out


def bet_board_rows(
    matchups: List[Dict[str, Any]],
    *,
    min_edge: float,
    min_completeness: float,
) -> List[Dict[str, Any]]:
    """Compact rows for Bet Candidates summary table."""
    rows: List[Dict[str, Any]] = []
    for m in matchups:
        players = m.get("players") or []
        if len(players) != 2:
            continue
        deuce = m.get("deuce") or {}
        route = str(deuce.get("route") or m.get("prediction_source") or "")
        e0 = players[0].get("edge")
        e1 = players[1].get("edge")
        ae = 0.0
        if e0 is not None:
            ae = max(ae, abs(float(e0)))
        if e1 is not None:
            ae = max(ae, abs(float(e1)))
        edge = float(e0) if e0 is not None and (e1 is None or abs(float(e0)) >= abs(float(e1))) else float(e1 or 0.0)
        comp = deuce.get("data_completeness")
        comp_f = float(comp) if comp is not None else None
        rec = decision_record(
            edge=edge,
            abs_edge=ae,
            completeness=comp_f,
            market_quality=str(m.get("market_quality") or ""),
            route=route,
            edge_reliable=bool(m.get("edge_reliable", False)),
            low_information=bool(m.get("low_information_prediction", False)),
            market_valid=bool(m.get("market_valid", True)),
            low_liquidity=bool(m.get("low_liquidity", False)),
            min_edge=min_edge,
            min_completeness=min_completeness,
        )
        na = players[0].get("name", "")
        nb = players[1].get("name", "")
        ev = str(m.get("event") or "")[:48]
        info_line = str(m.get("event_info_line") or "")[:28]
        phase = str(m.get("market_phase_label") or "")[:12]
        rows.append(
            {
                "Event": ev,
                "Event info": info_line or "—",
                "Phase": phase or "—",
                "Matchup": f"{na} vs {nb}",
                "Score": round(board_rank_score(m), 4),
                "|edge|": ae,
                "Route": route,
                "Compl.": comp_f,
                "Mkt Q": str(m.get("market_quality") or "—"),
                "Action": rec.final_action,
                "Edge": UI_PASS if rec.edge_pass else UI_FAIL,
                "Comp": UI_PASS if rec.completeness_pass else UI_FAIL,
                "MQ": UI_PASS if rec.market_quality_pass else UI_WARN,
            }
        )
    return rows


def route_counts(matchups: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for m in matchups:
        deuce = m.get("deuce") or {}
        r = str(deuce.get("route") or m.get("prediction_source") or "unknown")
        c[r] = c.get(r, 0) + 1
    return c
