"""
Feature engineering for the tennis match prediction model (refactored).
- Two-model design: Core (pre-match, broad coverage) and Enhanced (Core + ATP leaderboard when coverage).
- Score-derived features are LEAKAGE for pre-match: never used in Core/Enhanced.
- Matchup deltas with median imputation and missingness indicators (never treat missing as 0).
"""
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, List, Literal, Optional, Tuple

from deuce.name_utils import normalize_player_name

from deuce.features.base_features import DEFAULT_MEDIANS, PlayerStats
from deuce.features.coverage_features import (
    confidence_from_completeness,
    has_atp_leaderboard_coverage,
    leaderboard_coverage_score,
    should_route_enhanced,
)
from deuce.features.interaction_features import INTERACTION_FEATURES
from deuce.features.tier_lists import (
    CORE_FEATURE_NAMES,
    ENHANCED_EXTRA_FEATURE_NAMES,
    ENHANCED_FEATURE_NAMES,
    LEAKAGE_RISK_FEATURES,
)

# Back-compat alias used in docs / older scripts
LEAKAGE_RISK_FEATURES_LIST: List[str] = LEAKAGE_RISK_FEATURES

# Feature names in fixed order: base deltas + missing flags, surface/env, interactions, recent form + missing.
# Rank/points appear first but are downweighted via scaling and prior-only usage.
FEATURE_NAMES: List[str] = [
    "rank_diff",
    "rank_diff_missing",
    "log_points_diff",
    "log_points_diff_missing",
    "elo_diff",
    "elo_diff_missing",
    "surface_elo_diff",
    "surface_elo_diff_missing",
    "serve_diff",
    "serve_diff_missing",
    "return_diff",
    "return_diff_missing",
    "pressure_diff",
    "pressure_diff_missing",
    "winloss_diff",
    "winloss_diff_missing",
    "tdi_diff",
    "tdi_diff_missing",
    "surface_speed_rating",
    "indoor_flag",
    "altitude_factor",
    "climate_stress_index",
    "round_importance",
    "court_type_indoor",
    "tournament_weight",
    "int_serve_surface_speed",
    "int_return_slow_surface",
    "int_pressure_round",
    "int_serve_altitude",
    "int_return_climate",
    "last5_win_pct_diff",
    "last5_win_pct_missing",
    "last10_win_pct_diff",
    "last10_win_pct_missing",
    "recent_hold_diff",
    "recent_hold_missing",
    "recent_break_diff",
    "recent_break_missing",
    "deciding_set_diff",
    "deciding_set_missing",
    "minutes_last_7_days_diff",
    "minutes_last_7_days_missing",
    "days_since_last_match_diff",
    "days_since_last_match_missing",
    "timezone_shift_diff",
    "timezone_shift_missing",
    "travel_distance_diff",
    "travel_distance_missing",
    # Match context (best_of, draw_size: no missing flags; default if absent)
    "best_of",
    "draw_size",
    # Match-level / demographic (with missing flags)
    "age_diff",
    "age_diff_missing",
    "height_diff",
    "height_diff_missing",
    "hand_lefty_matchup",
    "hand_lefty_missing",
    "ace_rate_diff",
    "ace_rate_diff_missing",
    "df_rate_diff",
    "df_rate_diff_missing",
    "first_serve_in_pct_diff",
    "first_serve_in_pct_diff_missing",
    "first_serve_won_pct_diff",
    "first_serve_won_pct_diff_missing",
    "second_serve_won_pct_diff",
    "second_serve_won_pct_diff_missing",
    "service_games_won_pct_diff",
    "service_games_won_pct_diff_missing",
    "bp_saved_pct_diff",
    "bp_saved_pct_diff_missing",
    "bp_faced_pct_diff",
    "bp_faced_pct_diff_missing",
    "match_minutes",
    "match_minutes_missing",
    "deciding_set_flag",
    "deciding_set_flag_missing",
    "int_lefty_surface",
    # --- Priority upgrade (v3): seed, tourney_level, entry, event_id, score, nationality, match_num ---
    "seed_diff",
    "seed_diff_missing",
    "seeded_flag_a",
    "seeded_flag_a_missing",
    "seeded_flag_b",
    "seeded_flag_b_missing",
    "both_seeded_flag",
    "unseeded_vs_seeded_flag",
    "slam_flag",
    "masters_flag",
    "atp500_flag",
    "atp250_flag",
    "challenger_flag",
    "qualifier_flag_a",
    "qualifier_flag_b",
    "wildcard_flag_a",
    "wildcard_flag_b",
    "lucky_loser_flag_a",
    "lucky_loser_flag_b",
    "special_exempt_flag_a",
    "special_exempt_flag_b",
    "qualifier_vs_direct_flag",
    "wildcard_vs_seeded_flag",
    "entry_missing_a",
    "entry_missing_b",
    "event_id_hash",
    "event_id_missing",
    "sets_won_a",
    "sets_won_a_missing",
    "sets_won_b",
    "sets_won_b_missing",
    "straight_sets_flag",
    "straight_sets_flag_missing",
    "tiebreak_count",
    "tiebreak_count_missing",
    "retirement_walkover_flag",
    "long_match_flag",
    "long_match_flag_missing",
    "comeback_flag",
    "comeback_flag_missing",
    "home_flag_a",
    "home_flag_a_missing",
    "home_flag_b",
    "home_flag_b_missing",
    "same_country_flag",
    "home_advantage_diff",
    "home_advantage_diff_missing",
    "match_num",
    "match_num_missing",
    "in_event_fatigue_proxy",
]

# Groups for contribution breakdown and rank-dominance check
RANK_POINTS_FEATURES = ["rank_diff", "log_points_diff"]
# MATCHUP_FEATURES previously contained ATP leaderboard / Elo deltas.
# These are no longer part of the modeling feature space; keep the alias
# for compatibility but leave it empty so no bucket depends on them.
MATCHUP_FEATURES: List[str] = []
CONTEXT_FEATURES = [
    "surface_speed_rating", "indoor_flag", "altitude_factor", "climate_stress_index",
    "round_importance", "court_type_indoor", "tournament_weight", "best_of", "draw_size",
]
MATCH_LEVEL_FEATURES = [
    "age_diff", "height_diff", "hand_lefty_matchup",
    "ace_rate_diff", "df_rate_diff", "first_serve_in_pct_diff", "first_serve_won_pct_diff",
    "second_serve_won_pct_diff", "service_games_won_pct_diff", "bp_saved_pct_diff", "bp_faced_pct_diff",
    "match_minutes", "deciding_set_flag", "int_lefty_surface",
]
RECENT_FORM_FEATURES = ["last5_win_pct_diff", "last10_win_pct_diff", "recent_hold_diff", "recent_break_diff", "deciding_set_diff",
                        "minutes_last_7_days_diff", "days_since_last_match_diff", "timezone_shift_diff", "travel_distance_diff"]
# v3 priority upgrade buckets (for explanation / Streamlit)
DRAW_STATUS_FEATURES = ["seed_diff", "seeded_flag_a", "seeded_flag_b", "both_seeded_flag", "unseeded_vs_seeded_flag"]
EVENT_LEVEL_FEATURES = ["slam_flag", "masters_flag", "atp500_flag", "atp250_flag", "challenger_flag"]
ENTRY_PATH_FEATURES = [
    "qualifier_flag_a", "qualifier_flag_b", "wildcard_flag_a", "wildcard_flag_b",
    "lucky_loser_flag_a", "lucky_loser_flag_b", "special_exempt_flag_a", "special_exempt_flag_b",
    "qualifier_vs_direct_flag", "wildcard_vs_seeded_flag",
]
EVENT_IDENTITY_FEATURES = ["event_id_hash"]
SCORE_CONTEXT_FEATURES = [
    "sets_won_a", "sets_won_b", "straight_sets_flag", "tiebreak_count",
    "retirement_walkover_flag", "long_match_flag", "comeback_flag",
]
NATIONALITY_HOME_FEATURES = ["home_flag_a", "home_flag_b", "same_country_flag", "home_advantage_diff"]
MATCH_PROGRESSION_FEATURES = ["match_num", "in_event_fatigue_proxy"]


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert value to float; return default for None, empty, or invalid."""
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return default
        return float(value)
    try:
        s = str(value).strip().replace(",", "")
        if not s:
            return default
        return float(s)
    except (ValueError, TypeError):
        return default


def _present(v: Optional[float]) -> bool:
    return v is not None and (isinstance(v, float) or isinstance(v, int))


SURFACE_CLAY = "clay"
SURFACE_GRASS = "grass"
SURFACE_HARD_OUTDOOR = "hard_outdoor"
SURFACE_HARD_INDOOR = "hard_indoor"


@dataclass
class MatchFeatures:
    """Feature vector plus metadata for confidence and warnings."""
    feature_vector: List[float] = field(default_factory=list)
    feature_dict: Dict[str, float] = field(default_factory=dict)
    n_missing: int = 0
    n_total: int = 0
    rank_points_contribution: float = 0.0  # set by caller from model coefs
    is_rank_dominant: bool = False


def _delta_impute(
    a: Optional[float],
    b: Optional[float],
    median_diff: float,
) -> Tuple[float, int]:
    """
    Return (diff_value, missing_flag). Never use 0 for missing; use median_diff and set flag=1.
    """
    if _present(a) and _present(b):
        return (float(a) - float(b), 0)
    return (median_diff, 1)


def _log_points_diff(a: Optional[float], b: Optional[float], median_diff: float) -> Tuple[float, int]:
    if _present(a) and _present(b):
        la = math.log1p(max(0, float(a)))
        lb = math.log1p(max(0, float(b)))
        return (la - lb, 0)
    return (median_diff, 1)


def _surface_context(context: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Extract surface/environment scalars from context. No missing → sensible defaults."""
    out = {
        "surface_speed_rating": 5.0,
        "indoor_flag": 0.0,
        "altitude_factor": 0.0,
        "climate_stress_index": 2.5,
        "round_importance": 1.0,
        "court_type_indoor": 0.0,
        "tournament_weight": 1.0,
        "best_of": 3.0,
        "draw_size": 64.0,
    }
    if not context:
        return out
    surface = (context.get("surface") or "").strip().lower()
    speed = safe_float(context.get("surface_speed_rating"), None)
    if speed is not None:
        out["surface_speed_rating"] = max(0.0, min(10.0, float(speed)))
    else:
        if surface == "clay":
            out["surface_speed_rating"] = 3.5
        elif surface == "grass":
            out["surface_speed_rating"] = 4.0
        elif surface in ("hard_outdoor", "hard_indoor"):
            out["surface_speed_rating"] = 5.0
    ct = (context.get("court_type") or "").strip().lower()
    if ct == "indoor" or surface == "hard_indoor":
        out["indoor_flag"] = 1.0
        out["court_type_indoor"] = 1.0
    alt_m = safe_float(context.get("altitude_m"), None)
    if alt_m is not None:
        out["altitude_factor"] = min(alt_m / 1000.0, 1.0)
    temp = safe_float(context.get("temperature_avg"), None)
    hum = safe_float(context.get("humidity_avg"), None)
    if temp is not None or hum is not None:
        t = temp if temp is not None else 70.0
        h = hum if hum is not None else 50.0
        out["climate_stress_index"] = (t + 0.1 * h) / 10.0
    ri = context.get("round_importance")
    if ri is not None:
        out["round_importance"] = max(1.0, min(2.0, float(ri)))
    elif context.get("round"):
        rom = context.get("round_importance_map") or {}
        r = (context.get("round") or "").strip().upper()
        out["round_importance"] = float(rom.get(r, 1.0)) if rom else 1.0
    w = context.get("weight") or context.get("tournament_weight")
    if w is not None:
        out["tournament_weight"] = max(0.1, min(5.0, float(w)))
    bo = context.get("best_of")
    if bo is not None:
        try:
            out["best_of"] = float(max(3, min(5, int(bo))))
        except (TypeError, ValueError):
            pass
    ds = context.get("draw_size")
    if ds is not None:
        try:
            out["draw_size"] = max(8.0, min(256.0, float(ds)))
        except (TypeError, ValueError):
            pass
    return out


def _rate_or_none(num: Optional[float], den: Optional[float]) -> Optional[float]:
    """Return num/den as float if both present and den > 0, else None."""
    if num is None or den is None or not (isinstance(den, (int, float)) and float(den) > 0):
        return None
    try:
        return float(num) / float(den)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _parse_score(score_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse score string to sets won per side, straight sets, tiebreaks, retirement, comeback.
    Returns dict with sets_winner, sets_loser, straight_sets, tiebreak_count, retirement, walkover, comeback.
    Score format: "6-4 6-4" or "6-7(2) 7-6(3) 7-5". Winner is first player in each set.
    """
    out = {
        "sets_winner": None,
        "sets_loser": None,
        "straight_sets": None,
        "tiebreak_count": 0,
        "retirement": False,
        "walkover": False,
        "comeback": False,
    }
    if not score_str or not isinstance(score_str, str):
        return out
    s = score_str.strip().upper()
    if "W/O" in s or "WALKOVER" in s or "WO" == s:
        out["walkover"] = True
        return out
    if "RET" in s or "RETIRED" in s:
        out["retirement"] = True
    parts = s.replace("(", " ").replace(")", " ").split()
    set_scores: List[Tuple[int, int]] = []
    for i, token in enumerate(parts):
        if "-" in token:
            try:
                a, b = token.split("-")
                set_scores.append((int(a.strip()), int(b.strip())))
            except (ValueError, TypeError):
                pass
    if not set_scores:
        return out
    sets_winner = sum(1 for a, b in set_scores if a > b)
    sets_loser = sum(1 for a, b in set_scores if b > a)
    out["sets_winner"] = sets_winner
    out["sets_loser"] = sets_loser
    out["straight_sets"] = 1.0 if (sets_winner >= 2 and sets_loser == 0) or (sets_loser >= 2 and sets_winner == 0) else 0.0
    out["tiebreak_count"] = sum(1 for a, b in set_scores if (a == 7 and b >= 5) or (b == 7 and a >= 5))
    if len(set_scores) >= 2 and sets_winner >= 1 and sets_loser >= 1:
        first_set_winner_won = set_scores[0][0] > set_scores[0][1]
        if not first_set_winner_won:
            out["comeback"] = True
    return out


def _tourney_level_flags(tier_or_level: Optional[str]) -> Dict[str, float]:
    """Map tier/tourney_level string to binary flags. Unknown -> all 0."""
    t = (tier_or_level or "").strip().upper()
    out = {"slam": 0.0, "masters": 0.0, "atp500": 0.0, "atp250": 0.0, "challenger": 0.0}
    if "G" == t or "GRAND SLAM" in t or "SLAM" in t:
        out["slam"] = 1.0
    elif "M" == t or "MASTERS" in t or "1000" in t:
        out["masters"] = 1.0
    elif "500" in t:
        out["atp500"] = 1.0
    elif "250" in t or "A" == t:
        out["atp250"] = 1.0
    elif "CH" in t or "CHALLENGER" in t:
        out["challenger"] = 1.0
    return out


def _entry_flags(entry_str: Optional[str]) -> Dict[str, float]:
    """Map entry code (Q, WC, LL, SE, etc.) to qualifier, wildcard, lucky_loser, special_exempt. Missing -> all 0."""
    e = (entry_str or "").strip().upper()
    out = {"qualifier": 0.0, "wildcard": 0.0, "lucky_loser": 0.0, "special_exempt": 0.0}
    if not e or e in ("", "ND", "DIRECT"):
        return out
    if "Q" in e or e == "Q":
        out["qualifier"] = 1.0
    if "WC" in e or e == "WC":
        out["wildcard"] = 1.0
    if "LL" in e or e == "LL":
        out["lucky_loser"] = 1.0
    if "SE" in e or e == "SE":
        out["special_exempt"] = 1.0
    return out


def _event_id_hash(tourney_id: Optional[str], event_id: Optional[str]) -> Tuple[float, int]:
    """Return (hash in [0,1], missing). Uses tourney_id or event_id. Stable hash for same id."""
    raw = (tourney_id or event_id or "").strip()
    if not raw:
        return (0.5, 1)
    h = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16) % (2**31)
    return (float(h / (2**31)), 0)


def parse_score(score_str: Optional[str]) -> Dict[str, Any]:
    """Public helper: parse score string to sets_winner, sets_loser, straight_sets, tiebreak_count, retirement, walkover, comeback. Used by loaders."""
    return _parse_score(score_str)


def _home_flags(
    ioc_a: Optional[str],
    ioc_b: Optional[str],
    tournament_country: Optional[str],
) -> Dict[str, Any]:
    """
    Return home_flag_a, home_flag_b, same_country_flag, home_advantage_diff.
    tournament_country: ISO or name e.g. USA, Spain. Conservative: if ambiguous, home=0.
    """
    out = {"home_flag_a": 0.0, "home_flag_b": 0.0, "same_country_flag": 0.0, "home_advantage_diff": 0.0}
    a = (ioc_a or "").strip().upper()[:3]
    b = (ioc_b or "").strip().upper()[:3]
    tc = (tournament_country or "").strip().upper()[:3]
    if not tc:
        return out
    home_a = 1.0 if a and tc and a == tc else 0.0
    home_b = 1.0 if b and tc and b == tc else 0.0
    out["home_flag_a"] = home_a
    out["home_flag_b"] = home_b
    out["same_country_flag"] = 1.0 if a and b and a == b else 0.0
    out["home_advantage_diff"] = home_a - home_b
    return out


def build_match_features(
    player_a: PlayerStats,
    player_b: PlayerStats,
    tournament_context: Optional[Dict[str, Any]] = None,
    medians: Optional[Dict[str, float]] = None,
    match_context: Optional[Dict[str, Any]] = None,
    feature_set: Literal["core", "enhanced", "legacy"] = "legacy",
) -> MatchFeatures:
    """
    Build feature vector with imputation and missingness indicators.
    feature_set: "core" = pre-match only, no score-derived, no ATP leaderboard;
                 "enhanced" = core + ATP leaderboard + interactions;
                 "legacy" = full 127-feature set (includes score-derived; do not use for pre-match).
    When feature_set is "core" or "enhanced", score/sets_won/outcome from match_context are
    ignored (no target leakage).
    """
    med = {**DEFAULT_MEDIANS, **(medians or {})}
    mc = match_context or {}
    pre_match = feature_set in ("core", "enhanced")

    # Merge match_context into context so _surface_context sees best_of, draw_size
    ctx = dict(tournament_context or {})
    if mc.get("best_of") is not None:
        ctx["best_of"] = mc["best_of"]
    if mc.get("draw_size") is not None:
        ctx["draw_size"] = mc["draw_size"]

    # rank_diff = rank_B - rank_A so positive => A better (lower rank number)
    rank_diff, rank_miss = _delta_impute(player_b.rank, player_a.rank, med["rank_diff"])
    log_pts, log_pts_miss = _log_points_diff(player_a.points, player_b.points, med["log_points_diff"])
    elo_d, elo_miss = _delta_impute(player_a.elo, player_b.elo, med["elo_diff"])
    surf_elo_d, surf_elo_miss = _delta_impute(player_a.surface_elo, player_b.surface_elo, med["surface_elo_diff"])
    serve_d, serve_miss = _delta_impute(player_a.serve_rating, player_b.serve_rating, med["serve_diff"])
    return_d, return_miss = _delta_impute(player_a.return_rating, player_b.return_rating, med["return_diff"])
    pressure_d, pressure_miss = _delta_impute(player_a.pressure_rating, player_b.pressure_rating, med["pressure_diff"])
    winloss_d, winloss_miss = _delta_impute(player_a.winloss_index, player_b.winloss_index, med["winloss_diff"])
    tdi_d, tdi_miss = _delta_impute(player_a.tdi_perf, player_b.tdi_perf, med["tdi_diff"])

    surf = _surface_context(ctx)
    spd = surf["surface_speed_rating"]
    alt = surf["altitude_factor"]
    climate = surf["climate_stress_index"]
    rnd = surf["round_importance"]

    # Interactions (use imputed deltas)
    int_serve_speed = serve_d * spd
    int_return_slow = return_d * (11.0 - spd)
    int_pressure_rnd = pressure_d * rnd
    int_serve_alt = serve_d * alt
    int_return_climate = return_d * climate

    last5_d, last5_m = _delta_impute(player_a.last5_win_pct, player_b.last5_win_pct, med["last5_win_pct_diff"])
    last10_d, last10_m = _delta_impute(player_a.last10_win_pct, player_b.last10_win_pct, med["last10_win_pct_diff"])
    hold_d, hold_m = _delta_impute(player_a.recent_hold_pct, player_b.recent_hold_pct, med["recent_hold_diff"])
    break_d, break_m = _delta_impute(player_a.recent_break_pct, player_b.recent_break_pct, med["recent_break_diff"])
    dec_d, dec_m = _delta_impute(player_a.deciding_set_pct, player_b.deciding_set_pct, med["deciding_set_diff"])
    min7_d, min7_m = _delta_impute(player_a.minutes_last_7_days, player_b.minutes_last_7_days, med["minutes_last_7_days_diff"])
    days_d, days_m = _delta_impute(player_a.days_since_last_match, player_b.days_since_last_match, med["days_since_last_match_diff"])
    tz_d, tz_m = _delta_impute(player_a.timezone_shift, player_b.timezone_shift, med["timezone_shift_diff"])
    travel_d, travel_m = _delta_impute(player_a.travel_distance, player_b.travel_distance, med["travel_distance_diff"])

    # best_of, draw_size from surf (from context / match_context)
    best_of = surf.get("best_of", 3.0)
    draw_size = surf.get("draw_size", 64.0)

    # Match-level / demographic from match_context
    age_a = safe_float(mc.get("age_a"))
    age_b = safe_float(mc.get("age_b"))
    age_diff, age_miss = _delta_impute(age_a, age_b, med["age_diff"])
    ht_a = safe_float(mc.get("height_a") or mc.get("ht_a"))
    ht_b = safe_float(mc.get("height_b") or mc.get("ht_b"))
    height_diff, height_miss = _delta_impute(ht_a, ht_b, med["height_diff"])
    hand_a = (str(mc.get("hand_a") or "").strip().upper())[:1]
    hand_b = (str(mc.get("hand_b") or "").strip().upper())[:1]
    if hand_a and hand_b:
        lefty_matchup = 1.0 if (hand_a == "L" and hand_b != "L") or (hand_b == "L" and hand_a != "L") else 0.0
        hand_miss = 0
    else:
        lefty_matchup = 0.0
        hand_miss = 1
    int_lefty_surf = lefty_matchup * spd

    # Serve/return rates from match-level stats (TML columns)
    svpt_a = safe_float(mc.get("svpt_a") or mc.get("w_svpt"))
    svpt_b = safe_float(mc.get("svpt_b") or mc.get("l_svpt"))
    ace_a = safe_float(mc.get("ace_a") or mc.get("w_ace"))
    ace_b = safe_float(mc.get("ace_b") or mc.get("l_ace"))
    df_a = safe_float(mc.get("df_a") or mc.get("w_df"))
    df_b = safe_float(mc.get("df_b") or mc.get("l_df"))
    first_in_a = safe_float(mc.get("1stIn_a") or mc.get("w_1stIn"))
    first_in_b = safe_float(mc.get("1stIn_b") or mc.get("l_1stIn"))
    first_won_a = safe_float(mc.get("1stWon_a") or mc.get("w_1stWon"))
    first_won_b = safe_float(mc.get("1stWon_b") or mc.get("l_1stWon"))
    second_won_a = safe_float(mc.get("2ndWon_a") or mc.get("w_2ndWon"))
    second_won_b = safe_float(mc.get("2ndWon_b") or mc.get("l_2ndWon"))
    sv_gms_a = safe_float(mc.get("SvGms_a") or mc.get("w_SvGms"))
    sv_gms_b = safe_float(mc.get("SvGms_b") or mc.get("l_SvGms"))
    bp_saved_a = safe_float(mc.get("bpSaved_a") or mc.get("w_bpSaved"))
    bp_saved_b = safe_float(mc.get("bpSaved_b") or mc.get("l_bpSaved"))
    bp_faced_a = safe_float(mc.get("bpFaced_a") or mc.get("w_bpFaced"))
    bp_faced_b = safe_float(mc.get("bpFaced_b") or mc.get("l_bpFaced"))

    ace_rate_a = _rate_or_none(ace_a, svpt_a)
    ace_rate_b = _rate_or_none(ace_b, svpt_b)
    ace_rate_d, ace_rate_m = _delta_impute(ace_rate_a, ace_rate_b, med["ace_rate_diff"])
    df_rate_a = _rate_or_none(df_a, svpt_a)
    df_rate_b = _rate_or_none(df_b, svpt_b)
    df_rate_d, df_rate_m = _delta_impute(df_rate_a, df_rate_b, med["df_rate_diff"])
    first_in_pct_a = _rate_or_none(first_in_a, svpt_a)
    first_in_pct_b = _rate_or_none(first_in_b, svpt_b)
    first_in_pct_d, first_in_pct_miss = _delta_impute(first_in_pct_a, first_in_pct_b, med["first_serve_in_pct_diff"])
    first_won_pct_a = _rate_or_none(first_won_a, first_in_a) if first_in_a and first_in_a > 0 else None
    first_won_pct_b = _rate_or_none(first_won_b, first_in_b) if first_in_b and first_in_b > 0 else None
    first_won_pct_d, first_won_pct_miss = _delta_impute(first_won_pct_a, first_won_pct_b, med["first_serve_won_pct_diff"])
    second_won_pct_d, second_won_pct_miss = (0.0, 1)
    if second_won_a is not None and second_won_b is not None and svpt_a and svpt_b:
        second_svpt_a = (svpt_a - first_in_a) if first_in_a is not None else None
        second_svpt_b = (svpt_b - first_in_b) if first_in_b is not None else None
        sw_a = _rate_or_none(second_won_a, second_svpt_a) if second_svpt_a and second_svpt_a > 0 else None
        sw_b = _rate_or_none(second_won_b, second_svpt_b) if second_svpt_b and second_svpt_b > 0 else None
        second_won_pct_d, second_won_pct_miss = _delta_impute(sw_a, sw_b, med["second_serve_won_pct_diff"])
    sv_gms_pct_a = _rate_or_none(sv_gms_a, svpt_a) if svpt_a and svpt_a > 0 else None
    sv_gms_pct_b = _rate_or_none(sv_gms_b, svpt_b) if svpt_b and svpt_b > 0 else None
    service_gms_d, service_gms_miss = _delta_impute(sv_gms_pct_a, sv_gms_pct_b, med["service_games_won_pct_diff"])
    bp_saved_pct_a = _rate_or_none(bp_saved_a, bp_faced_a) if bp_faced_a and bp_faced_a > 0 else None
    bp_saved_pct_b = _rate_or_none(bp_saved_b, bp_faced_b) if bp_faced_b and bp_faced_b > 0 else None
    bp_saved_d, bp_saved_miss = _delta_impute(bp_saved_pct_a, bp_saved_pct_b, med["bp_saved_pct_diff"])
    bp_faced_pct_a = _rate_or_none(bp_faced_a, svpt_a) if svpt_a and svpt_a > 0 else None
    bp_faced_pct_b = _rate_or_none(bp_faced_b, svpt_b) if svpt_b and svpt_b > 0 else None
    bp_faced_d, bp_faced_miss = _delta_impute(bp_faced_pct_a, bp_faced_pct_b, med["bp_faced_pct_diff"])

    minutes_val = safe_float(mc.get("minutes") or mc.get("match_minutes"))
    if minutes_val is not None:
        match_minutes, match_minutes_miss = float(minutes_val), 0
    else:
        match_minutes, match_minutes_miss = med["match_minutes"], 1
    deciding_set_val = mc.get("deciding_set")
    if deciding_set_val is not None:
        deciding_set_flag = 1.0 if deciding_set_val in (True, 1, "1", "yes") else 0.0
        deciding_set_miss = 0
    else:
        deciding_set_flag = 0.0
        deciding_set_miss = 1

    # --- Seed (v3): seed_diff, seeded flags; missing != 0 ---
    seed_a = safe_float(mc.get("seed_a"))
    seed_b = safe_float(mc.get("seed_b"))
    if seed_a is not None and seed_a > 0:
        seed_a = min(seed_a, 33.0)
    if seed_b is not None and seed_b > 0:
        seed_b = min(seed_b, 33.0)
    seed_diff, seed_diff_miss = _delta_impute(seed_b, seed_a, med["seed_diff"])
    seeded_a = 1.0 if (seed_a is not None and 1 <= seed_a <= 32) else 0.0
    seeded_a_miss = 1 if seed_a is None else 0
    seeded_b = 1.0 if (seed_b is not None and 1 <= seed_b <= 32) else 0.0
    seeded_b_miss = 1 if seed_b is None else 0
    both_seeded = 1.0 if seeded_a == 1.0 and seeded_b == 1.0 else 0.0
    unseeded_vs_seeded = 1.0 if (seeded_a != seeded_b) else 0.0

    # --- Tourney level (v3): from ctx or mc ---
    tier_str = ctx.get("tier") or mc.get("tourney_level") or ""
    level_flags = _tourney_level_flags(tier_str)
    slam_flag = level_flags["slam"]
    masters_flag = level_flags["masters"]
    atp500_flag = level_flags["atp500"]
    atp250_flag = level_flags["atp250"]
    challenger_flag = level_flags["challenger"]

    # --- Entry (v3): qualifier, WC, LL, SE; missing handled ---
    entry_a = (mc.get("entry_a") or mc.get("winner_entry") or "").strip()
    entry_b = (mc.get("entry_b") or mc.get("loser_entry") or "").strip()
    ea = _entry_flags(entry_a if entry_a else None)
    eb = _entry_flags(entry_b if entry_b else None)
    qualifier_flag_a = ea["qualifier"]
    qualifier_flag_b = eb["qualifier"]
    wildcard_flag_a = ea["wildcard"]
    wildcard_flag_b = eb["wildcard"]
    lucky_loser_flag_a = ea["lucky_loser"]
    lucky_loser_flag_b = eb["lucky_loser"]
    special_exempt_flag_a = ea["special_exempt"]
    special_exempt_flag_b = eb["special_exempt"]
    has_any_entry = mc.get("entry_a") is not None or mc.get("winner_entry") is not None or mc.get("loser_entry") is not None
    entry_missing_a = 0 if has_any_entry else 1
    entry_missing_b = 0 if has_any_entry else 1
    qualifier_vs_direct = 1.0 if (ea["qualifier"] != eb["qualifier"]) else 0.0
    wildcard_vs_seeded = 1.0 if (ea["wildcard"] == 1.0 or eb["wildcard"] == 1.0) and (seeded_a == 1.0 or seeded_b == 1.0) else 0.0

    # --- Event identity (v3): hash of tourney_id / event_id ---
    event_id_hash, event_id_miss = _event_id_hash(mc.get("tourney_id"), ctx.get("event_id"))

    # --- Score-derived (v3): LEAKAGE for pre-match — use only when feature_set == "legacy" ---
    if pre_match:
        sets_won_a, sets_won_a_miss = med["sets_won_a"], 1
        sets_won_b, sets_won_b_miss = med["sets_won_b"], 1
        straight_sets_flag, straight_sets_miss = 0.0, 1
        tiebreak_count, tiebreak_count_miss = med["tiebreak_count"], 1
        retirement_walkover_flag = 0.0
        long_match_flag, long_match_flag_miss = 0.0, 1
        comeback_flag, comeback_flag_miss = 0.0, 1
    else:
        score_str = mc.get("score")
        parsed = mc.get("score_parsed")
        if parsed is None and score_str:
            parsed = _parse_score(str(score_str))
        if parsed is None:
            parsed = {}
        sets_won_a_raw = safe_float(mc.get("sets_won_a"))
        sets_won_b_raw = safe_float(mc.get("sets_won_b"))
        if sets_won_a_raw is not None and sets_won_b_raw is not None:
            sets_won_a, sets_won_a_miss = float(sets_won_a_raw), 0
            sets_won_b, sets_won_b_miss = float(sets_won_b_raw), 0
        else:
            sets_won_a, sets_won_a_miss = med["sets_won_a"], 1
            sets_won_b, sets_won_b_miss = med["sets_won_b"], 1
        straight_sets = parsed.get("straight_sets")
        straight_sets_flag = float(straight_sets) if straight_sets is not None else 0.0
        straight_sets_miss = 1 if straight_sets is None and not score_str else 0
        tiebreak_count = parsed.get("tiebreak_count")
        if tiebreak_count is not None:
            tiebreak_count = float(tiebreak_count)
        else:
            tiebreak_count = med["tiebreak_count"]
        tiebreak_count_miss = 1 if (not score_str) else 0
        retirement_walkover_flag = 1.0 if parsed.get("retirement") or parsed.get("walkover") else 0.0
        minutes_val = safe_float(mc.get("minutes") or mc.get("match_minutes"))
        long_match_flag = 1.0 if (minutes_val is not None and minutes_val >= 180) else 0.0
        long_match_flag_miss = 1 if minutes_val is None else 0
        comeback_flag = 1.0 if parsed.get("comeback") else 0.0
        comeback_flag_miss = 1 if parsed.get("comeback") is None and not score_str else 0

    # --- Nationality / home (v3) ---
    ioc_a = (mc.get("ioc_a") or mc.get("winner_ioc") or "").strip()
    ioc_b = (mc.get("ioc_b") or mc.get("loser_ioc") or "").strip()
    tournament_country = (ctx.get("country") or mc.get("tournament_country") or "").strip()
    home = _home_flags(ioc_a or None, ioc_b or None, tournament_country or None)
    home_flag_a = home["home_flag_a"]
    home_flag_b = home["home_flag_b"]
    same_country_flag = home["same_country_flag"]
    home_advantage_diff = home["home_advantage_diff"]
    home_a_miss = 1 if not ioc_a and not mc.get("ioc_a") and not mc.get("winner_ioc") else 0
    home_b_miss = 1 if not ioc_b and not mc.get("ioc_b") and not mc.get("loser_ioc") else 0
    home_adv_miss = 1 if (home_a_miss and home_b_miss) else 0

    # --- Match progression (v3) ---
    match_num_val = safe_float(mc.get("match_num"))
    if match_num_val is not None:
        match_num_feat = float(match_num_val)
        match_num_miss = 0
    else:
        match_num_feat = med["match_num"]
        match_num_miss = 1
    in_event_fatigue = (match_num_feat * rnd) if (match_num_val is not None and rnd > 0) else med["in_event_fatigue_proxy"]

    vec = [
        rank_diff, float(rank_miss),
        log_pts, float(log_pts_miss),
        elo_d, float(elo_miss),
        surf_elo_d, float(surf_elo_miss),
        serve_d, float(serve_miss),
        return_d, float(return_miss),
        pressure_d, float(pressure_miss),
        winloss_d, float(winloss_miss),
        tdi_d, float(tdi_miss),
        spd, surf["indoor_flag"], alt, climate, rnd, surf["court_type_indoor"], surf["tournament_weight"],
        int_serve_speed, int_return_slow, int_pressure_rnd, int_serve_alt, int_return_climate,
        last5_d, float(last5_m), last10_d, float(last10_m),
        hold_d, float(hold_m), break_d, float(break_m), dec_d, float(dec_m),
        min7_d, float(min7_m), days_d, float(days_m), tz_d, float(tz_m), travel_d, float(travel_m),
        best_of, draw_size,
        age_diff, float(age_miss), height_diff, float(height_miss), lefty_matchup, float(hand_miss),
        ace_rate_d, float(ace_rate_m), df_rate_d, float(df_rate_m),
        first_in_pct_d, float(first_in_pct_miss), first_won_pct_d, float(first_won_pct_miss),
        second_won_pct_d, float(second_won_pct_miss), service_gms_d, float(service_gms_miss),
        bp_saved_d, float(bp_saved_miss), bp_faced_d, float(bp_faced_miss),
        match_minutes, float(match_minutes_miss), deciding_set_flag, float(deciding_set_miss),
        int_lefty_surf,
        seed_diff, float(seed_diff_miss), seeded_a, float(seeded_a_miss), seeded_b, float(seeded_b_miss),
        both_seeded, unseeded_vs_seeded,
        slam_flag, masters_flag, atp500_flag, atp250_flag, challenger_flag,
        qualifier_flag_a, qualifier_flag_b, wildcard_flag_a, wildcard_flag_b,
        lucky_loser_flag_a, lucky_loser_flag_b, special_exempt_flag_a, special_exempt_flag_b,
        qualifier_vs_direct, wildcard_vs_seeded, float(entry_missing_a), float(entry_missing_b),
        event_id_hash, float(event_id_miss),
        sets_won_a, float(sets_won_a_miss), sets_won_b, float(sets_won_b_miss),
        straight_sets_flag, float(straight_sets_miss), tiebreak_count, float(tiebreak_count_miss),
        retirement_walkover_flag, long_match_flag, float(long_match_flag_miss), comeback_flag, float(comeback_flag_miss),
        home_flag_a, float(home_a_miss), home_flag_b, float(home_b_miss),
        same_country_flag, home_advantage_diff, float(home_adv_miss),
        match_num_feat, float(match_num_miss), in_event_fatigue,
    ]
    fd = dict(zip(FEATURE_NAMES, vec))
    if feature_set == "core":
        names_out = CORE_FEATURE_NAMES
    elif feature_set == "enhanced":
        names_out = ENHANCED_FEATURE_NAMES
    else:
        names_out = FEATURE_NAMES
    vec_out = [fd.get(n, 0.0) for n in names_out]
    fd_out = {n: fd.get(n, 0.0) for n in names_out}
    n_miss = sum(1 for n in names_out if n.endswith("_missing") and fd_out.get(n, 0) == 1.0)
    return MatchFeatures(
        feature_vector=vec_out,
        feature_dict=fd_out,
        n_missing=n_miss,
        n_total=len(names_out),
    )


def compute_environment_score(context: Optional[Dict[str, Any]]) -> float:
    """Legacy composite environment score (kept for backward compatibility)."""
    if not context:
        return 0.0
    surface = (context.get("surface") or "").strip().lower()
    temp = safe_float(context.get("temperature_avg"), 0.0) or 0.0
    humidity = safe_float(context.get("humidity_avg"), 0.0) or 0.0
    altitude_m = safe_float(context.get("altitude_m"), 0.0) or 0.0
    surface_speed_rating = safe_float(context.get("surface_speed_rating"), None)
    court_type = (context.get("court_type") or "").strip().lower()
    if surface_speed_rating is None:
        surface_speed_rating = 5.0 if surface in ("hard_outdoor", "hard_indoor") else (3.5 if surface == "clay" else 4.0)
    if not court_type and surface == "hard_indoor":
        court_type = "indoor"
    if not court_type:
        court_type = "outdoor"
    climate_stress_index = temp + 0.1 * humidity
    altitude_factor = min(altitude_m / 1000.0, 1.0)
    indoor = 1.0 if court_type == "indoor" else 0.0
    return (
        0.35 * surface_speed_rating
        + 0.25 * altitude_factor * 10.0
        + 0.25 * (climate_stress_index / 10.0)
        + 0.15 * indoor
    )


def fallback_score(player_a: PlayerStats, player_b: PlayerStats) -> float:
    """
    Prior-only heuristic: rank and log_points with modest weight; no 0-fill for missing.
    Uses imputation so missing stats don't act as "tie".
    """
    med = DEFAULT_MEDIANS
    rank_diff, _ = _delta_impute(player_a.rank, player_b.rank, med["rank_diff"])
    log_pts, _ = _log_points_diff(player_a.points, player_b.points, med["log_points_diff"])
    serve_d, _ = _delta_impute(player_a.serve_rating, player_b.serve_rating, med["serve_diff"])
    return_d, _ = _delta_impute(player_a.return_rating, player_b.return_rating, med["return_diff"])
    # Prior: rank (negative = A better) and log points; then matchup if present
    rank_w = -0.008
    log_pts_w = 0.15
    serve_w = 0.03
    return_w = 0.03
    raw = rank_w * rank_diff + log_pts_w * log_pts + serve_w * serve_d + return_w * return_d
    try:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, raw))))
    except (OverflowError, ValueError):
        return 0.5


def feature_dict_from_match_features(mf: MatchFeatures) -> Dict[str, float]:
    """Return feature name -> value for inspection."""
    return dict(mf.feature_dict) if mf.feature_dict else {}


def _find_row_by_player(rows: List[Dict[str, Any]], player_name: str) -> Optional[Dict[str, Any]]:
    from deuce.name_utils import match_player
    if not rows:
        return None
    candidates = [r.get("player") or "" for r in rows if r.get("player")]
    matched = match_player(player_name, candidates)
    if not matched:
        return None
    for r in rows:
        if (r.get("player") or "").strip() == matched:
            return r
    return None


def build_player_stats_from_datasets(
    player_name: str,
    *,
    player_db: Optional[List[Dict[str, Any]]] = None,
    atp_rankings: Optional[List[Dict[str, Any]]] = None,
    serve_rows: Optional[List[Dict[str, Any]]] = None,
    return_rows: Optional[List[Dict[str, Any]]] = None,
    pressure_rows: Optional[List[Dict[str, Any]]] = None,
    winloss_rows: Optional[List[Dict[str, Any]]] = None,
    tdi_rows: Optional[List[Dict[str, Any]]] = None,
) -> PlayerStats:
    """Build PlayerStats from DB or ATP datasets. Missing values remain None (no 0)."""
    if player_db:
        row = _find_row_by_player(player_db, player_name)
        if row:
            return PlayerStats(
                name=player_name,
                rank=safe_float(row.get("rank")),
                points=safe_float(row.get("points")),
                elo=safe_float(row.get("elo")),
                surface_elo=safe_float(row.get("surface_elo")),
                serve_rating=safe_float(row.get("serve_rating")),
                return_rating=safe_float(row.get("return_rating")),
                pressure_rating=safe_float(row.get("pressure_rating")),
                winloss_index=safe_float(row.get("winloss_index")),
                tdi_perf=safe_float(row.get("tdi_perf")),
            )
    rank = points = None
    if atp_rankings:
        row = _find_row_by_player(atp_rankings, player_name)
        if row:
            rank = safe_float(row.get("rank"))
            points = safe_float(row.get("official points"))
    row_s = _find_row_by_player(serve_rows or [], player_name)
    serve_rating = safe_float(row_s.get("serve_rating")) if row_s else None
    row_r = _find_row_by_player(return_rows or [], player_name)
    return_rating = safe_float(row_r.get("return_rating")) if row_r else None
    row_p = _find_row_by_player(pressure_rows or [], player_name)
    pressure_rating = safe_float(row_p.get("pressure_rating")) if row_p else None
    row_w = _find_row_by_player(winloss_rows or [], player_name)
    winloss_index = safe_float(row_w.get("index")) if row_w else None
    row_t = _find_row_by_player(tdi_rows or [], player_name)
    tdi_perf = safe_float(row_t.get("performance_rating")) if row_t else None
    return PlayerStats(
        name=player_name,
        rank=rank,
        points=points,
        serve_rating=serve_rating,
        return_rating=return_rating,
        pressure_rating=pressure_rating,
        winloss_index=winloss_index,
        tdi_perf=tdi_perf,
    )
