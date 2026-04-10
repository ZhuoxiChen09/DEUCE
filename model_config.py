"""
Central config for the tennis win-probability model.
No magic numbers; single place to tune thresholds and feature groups.
Implementation plan: IMPLEMENTATION_PLAN_FEATURES.md.
"""
from typing import Dict, List

# Feature groups for contribution buckets (display and diagnostics).
# Order: base_strength (rank/points), tennis_skill, context, match_level, interactions, recent_form,
# v3: draw_status, event_level, entry_path, event_identity, score_context, nationality_home, match_progression, uncertainty_penalty.
CONTRIBUTION_BUCKETS = [
    "base_strength",
    "rank_points",
    "tennis_skill",
    "context",
    "match_level",
    "interactions",
    "recent_form",
    "draw_status",
    "event_level",
    "entry_path",
    "event_identity",
    "score_context",
    "nationality_home",
    "match_progression",
    "uncertainty_penalty",
]


def _bucket_features() -> Dict[str, List[str]]:
    from model_features import FEATURE_NAMES
    buckets: Dict[str, List[str]] = {
        "base_strength": ["rank_diff", "log_points_diff"],
        "rank_points": ["rank_diff", "log_points_diff"],
        # Leaderboard / Elo deltas appear in Enhanced tier bundles; Core leaves them empty.
        "tennis_skill": [
            "elo_diff", "surface_elo_diff",
            "serve_diff", "return_diff", "pressure_diff", "winloss_diff", "tdi_diff",
        ],
        "context": [
            "surface_speed_rating", "indoor_flag", "altitude_factor", "climate_stress_index",
            "round_importance", "court_type_indoor", "tournament_weight", "best_of", "draw_size",
        ],
        "match_level": [
            "age_diff", "height_diff", "hand_lefty_matchup",
            "ace_rate_diff", "df_rate_diff", "first_serve_in_pct_diff", "first_serve_won_pct_diff",
            "second_serve_won_pct_diff", "service_games_won_pct_diff", "bp_saved_pct_diff", "bp_faced_pct_diff",
            "match_minutes", "deciding_set_flag", "int_lefty_surface",
        ],
        "interactions": ["int_serve_surface_speed", "int_return_slow_surface", "int_pressure_round", "int_serve_altitude", "int_return_climate", "int_lefty_surface"],
        "recent_form": [
            "last5_win_pct_diff", "last10_win_pct_diff", "recent_hold_diff", "recent_break_diff",
            "deciding_set_diff", "minutes_last_7_days_diff", "days_since_last_match_diff",
            "timezone_shift_diff", "travel_distance_diff",
        ],
        "draw_status": ["seed_diff", "seeded_flag_a", "seeded_flag_b", "both_seeded_flag", "unseeded_vs_seeded_flag"],
        "event_level": ["slam_flag", "masters_flag", "atp500_flag", "atp250_flag", "challenger_flag"],
        "entry_path": [
            "qualifier_flag_a", "qualifier_flag_b", "wildcard_flag_a", "wildcard_flag_b",
            "lucky_loser_flag_a", "lucky_loser_flag_b", "special_exempt_flag_a", "special_exempt_flag_b",
            "qualifier_vs_direct_flag", "wildcard_vs_seeded_flag",
        ],
        "event_identity": ["event_id_hash"],
        "score_context": [
            "sets_won_a", "sets_won_b", "straight_sets_flag", "tiebreak_count",
            "retirement_walkover_flag", "long_match_flag", "comeback_flag",
        ],
        "nationality_home": ["home_flag_a", "home_flag_b", "same_country_flag", "home_advantage_diff"],
        "match_progression": ["match_num", "in_event_fatigue_proxy"],
        "uncertainty_penalty": [n for n in FEATURE_NAMES if n.endswith("_missing")],
    }
    return buckets


BUCKET_FEATURES: Dict[str, List[str]] = _bucket_features()

# Rank-dominance: if rank_points share of total |contribution| exceeds this, warn
RANK_POINTS_DOMINANCE_THRESHOLD = 0.5

# Elite-vs-elite: both players in top N → shrink probability toward 0.5
ELITE_RANK_THRESHOLD = 15
ELITE_SHRINK_STRENGTH = 0.3  # pull prob toward 0.5 by this factor

# Missing-data: if completeness below this, cap extreme probabilities
LOW_COMPLETENESS_THRESHOLD = 0.7
EXTREME_PROB_CAP_LO = 0.15
EXTREME_PROB_CAP_HI = 0.85

# Contribution display: treat |contrib| below this as effectively zero
CONTRIBUTION_ZERO_THRESHOLD = 1e-6
