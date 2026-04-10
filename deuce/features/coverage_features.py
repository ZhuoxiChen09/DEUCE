"""
ATP / WTA coverage scoring and Enhanced routing (no model imports).
"""

from __future__ import annotations

from deuce.features.base_features import PlayerStats


def has_atp_leaderboard_coverage(
    player_a: PlayerStats,
    player_b: PlayerStats,
    min_required: int = 2,
) -> bool:
    """
    True if both players have at least min_required of (serve_rating, return_rating, pressure_rating).
    """
    atp_keys = ("serve_rating", "return_rating", "pressure_rating")

    def count_present(p: PlayerStats) -> int:
        return sum(1 for k in atp_keys if getattr(p, k, None) is not None)

    return count_present(player_a) >= min_required and count_present(player_b) >= min_required


def leaderboard_coverage_score(player_a: PlayerStats, player_b: PlayerStats) -> float:
    """
    Average fraction of optional leaderboard stats present across both players, in [0, 1].
    """
    keys = ("serve_rating", "return_rating", "pressure_rating", "winloss_index", "tdi_perf")

    def frac(p: PlayerStats) -> float:
        return sum(1 for k in keys if getattr(p, k, None) is not None) / float(len(keys))

    return 0.5 * (frac(player_a) + frac(player_b))


def should_route_enhanced(
    player_a: PlayerStats,
    player_b: PlayerStats,
    *,
    min_strict_per_player: int = 2,
    min_coverage_score: float = 0.35,
) -> bool:
    """Enhanced routing: strict ATP coverage or sufficiently high blended coverage score."""
    if has_atp_leaderboard_coverage(player_a, player_b, min_required=min_strict_per_player):
        return True
    return leaderboard_coverage_score(player_a, player_b) >= min_coverage_score


def confidence_from_completeness(n_missing: int, n_total: int) -> float:
    """Feature completeness score in [0, 1]."""
    if n_total <= 0:
        return 0.0
    return max(0.0, 1.0 - (n_missing / n_total))
