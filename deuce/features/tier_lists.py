"""
Pre-match tier feature name lists (single source of truth for Core / Enhanced).

``model_features.py`` re-exports these for backward compatibility.
"""

from __future__ import annotations

from typing import List

LEAKAGE_RISK_FEATURES: List[str] = [
    "sets_won_a", "sets_won_a_missing", "sets_won_b", "sets_won_b_missing",
    "straight_sets_flag", "straight_sets_flag_missing",
    "tiebreak_count", "tiebreak_count_missing",
    "retirement_walkover_flag",
    "long_match_flag", "long_match_flag_missing",
    "comeback_flag", "comeback_flag_missing",
]

CORE_FEATURE_NAMES: List[str] = [
    "rank_diff", "rank_diff_missing",
    "log_points_diff", "log_points_diff_missing",
    "surface_speed_rating", "indoor_flag", "altitude_factor", "climate_stress_index",
    "round_importance", "court_type_indoor", "tournament_weight",
    "best_of", "draw_size",
    "age_diff", "age_diff_missing", "height_diff", "height_diff_missing",
    "hand_lefty_matchup", "hand_lefty_missing",
    "int_lefty_surface",
    "seed_diff", "seed_diff_missing",
    "seeded_flag_a", "seeded_flag_a_missing", "seeded_flag_b", "seeded_flag_b_missing",
    "both_seeded_flag", "unseeded_vs_seeded_flag",
    "slam_flag", "masters_flag", "atp500_flag", "atp250_flag", "challenger_flag",
    "qualifier_flag_a", "qualifier_flag_b", "wildcard_flag_a", "wildcard_flag_b",
    "lucky_loser_flag_a", "lucky_loser_flag_b", "special_exempt_flag_a", "special_exempt_flag_b",
    "qualifier_vs_direct_flag", "wildcard_vs_seeded_flag",
    "entry_missing_a", "entry_missing_b",
    "event_id_hash", "event_id_missing",
    "home_flag_a", "home_flag_a_missing", "home_flag_b", "home_flag_b_missing",
    "same_country_flag", "home_advantage_diff", "home_advantage_diff_missing",
    "match_num", "match_num_missing", "in_event_fatigue_proxy",
]

ENHANCED_EXTRA_FEATURE_NAMES: List[str] = [
    "elo_diff", "elo_diff_missing",
    "surface_elo_diff", "surface_elo_diff_missing",
    "serve_diff", "serve_diff_missing",
    "return_diff", "return_diff_missing",
    "pressure_diff", "pressure_diff_missing",
    "winloss_diff", "winloss_diff_missing",
    "tdi_diff", "tdi_diff_missing",
    "int_serve_surface_speed", "int_return_slow_surface", "int_pressure_round",
    "int_serve_altitude", "int_return_climate",
]

ENHANCED_FEATURE_NAMES: List[str] = CORE_FEATURE_NAMES + ENHANCED_EXTRA_FEATURE_NAMES
