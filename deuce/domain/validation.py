"""Validation rules for tennis match state."""

from __future__ import annotations

from deuce.domain.enums import MatchFormat, MatchStatus
from deuce.domain.models import MatchConfig, MatchState


def validate_match_state(state: MatchState, config: MatchConfig) -> list[str]:
    """Return validation errors (empty list = valid)."""
    errors: list[str] = []
    max_sets = config.sets_to_win * 2 - 1

    total_sets = state.sets_a + state.sets_b + len(
        [1 for s in [state.current_set] if s and not s.is_complete]
    )
    if state.sets_a + state.sets_b > max_sets:
        errors.append(
            f"Total completed sets {state.sets_a + state.sets_b} exceeds max {max_sets}."
        )

    if state.sets_a > config.sets_to_win:
        errors.append(f"Player A has {state.sets_a} sets but only needs {config.sets_to_win}.")
    if state.sets_b > config.sets_to_win:
        errors.append(f"Player B has {state.sets_b} sets but only needs {config.sets_to_win}.")

    if state.is_complete and state.current_set is not None:
        errors.append("Match is complete but current_set is not None.")
    if state.is_complete and state.current_game is not None:
        errors.append("Match is complete but current_game is not None.")

    for s in state.completed_sets:
        if not s.is_complete:
            errors.append(f"Set {s.set_number} in completed_sets but is_complete=False.")
        if s.winner is None:
            errors.append(f"Set {s.set_number} is complete but has no winner.")

    if state.current_set:
        cs = state.current_set
        if cs.games_a < 0 or cs.games_b < 0:
            errors.append(f"Negative game count in current set: {cs.games_a}-{cs.games_b}.")

    return errors


def validate_score_transition(
    before: MatchState, after: MatchState, config: MatchConfig
) -> list[str]:
    """Verify a state transition is legal."""
    errors: list[str] = []

    if after.version <= before.version:
        errors.append(
            f"Version did not advance: {before.version} -> {after.version}."
        )

    if after.sets_a < before.sets_a or after.sets_b < before.sets_b:
        errors.append("Sets cannot decrease.")

    if after.points_played_total < before.points_played_total:
        errors.append("Total points played cannot decrease.")

    if before.is_complete and not after.is_complete:
        errors.append("Match was complete but is no longer complete.")

    return errors
