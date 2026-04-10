"""
Core types and default imputation values (shared by ``model_features`` and training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from deuce.name_utils import normalize_player_name


# Default medians for imputation (training replaces with data-derived medians where fit).
DEFAULT_MEDIANS: dict[str, float] = {
    "rank_diff": 0.0,
    "log_points_diff": 0.0,
    "elo_diff": 0.0,
    "surface_elo_diff": 0.0,
    "serve_diff": 0.0,
    "return_diff": 0.0,
    "pressure_diff": 0.0,
    "winloss_diff": 0.0,
    "tdi_diff": 0.0,
    "last5_win_pct_diff": 0.0,
    "last10_win_pct_diff": 0.0,
    "recent_hold_diff": 0.0,
    "recent_break_diff": 0.0,
    "deciding_set_diff": 0.0,
    "minutes_last_7_days_diff": 0.0,
    "days_since_last_match_diff": 0.0,
    "timezone_shift_diff": 0.0,
    "travel_distance_diff": 0.0,
    "age_diff": 0.0,
    "height_diff": 0.0,
    "ace_rate_diff": 0.0,
    "df_rate_diff": 0.0,
    "first_serve_in_pct_diff": 0.0,
    "first_serve_won_pct_diff": 0.0,
    "second_serve_won_pct_diff": 0.0,
    "service_games_won_pct_diff": 0.0,
    "bp_saved_pct_diff": 0.0,
    "bp_faced_pct_diff": 0.0,
    "match_minutes": 90.0,
    "seed_diff": 0.0,
    "match_num": 5.0,
    "event_id_hash": 0.5,
    "sets_won_a": 2.0,
    "sets_won_b": 1.0,
    "tiebreak_count": 0.0,
    "home_advantage_diff": 0.0,
    "in_event_fatigue_proxy": 2.0,
}


@dataclass
class PlayerStats:
    """Player-level stats. All optional; missing means unknown (will be imputed + missing flag)."""

    name: str
    rank: Optional[float] = None
    points: Optional[float] = None
    elo: Optional[float] = None
    surface_elo: Optional[float] = None
    serve_rating: Optional[float] = None
    return_rating: Optional[float] = None
    pressure_rating: Optional[float] = None
    winloss_index: Optional[float] = None
    tdi_perf: Optional[float] = None
    last5_win_pct: Optional[float] = None
    last10_win_pct: Optional[float] = None
    recent_hold_pct: Optional[float] = None
    recent_break_pct: Optional[float] = None
    deciding_set_pct: Optional[float] = None
    minutes_last_7_days: Optional[float] = None
    days_since_last_match: Optional[float] = None
    timezone_shift: Optional[float] = None
    travel_distance: Optional[float] = None

    @property
    def normalized_name(self) -> str:
        return normalize_player_name(self.name)
