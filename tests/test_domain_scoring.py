"""Tests for the tennis scoring engine.

Covers: regular games, deuce/advantage, tiebreaks, set completion,
match completion (best-of-3), state reconstruction from scores,
and full match simulations.
"""

import pytest

from deuce.domain.enums import MatchFormat, MatchStatus, ServerSide, SetType
from deuce.domain.models import (
    GameState,
    MatchConfig,
    MatchState,
    Player,
    PointState,
    SetState,
    Tournament,
)
from deuce.domain.scoring import ScoringEngine
from deuce.domain.validation import validate_match_state


def _make_config(
    match_format: MatchFormat = MatchFormat.BEST_OF_3,
    final_set_tiebreak: bool = True,
) -> MatchConfig:
    return MatchConfig(
        match_id="test-001",
        tournament=Tournament(name="Test Open"),
        player_a=Player(player_id="a", name="Player A"),
        player_b=Player(player_id="b", name="Player B"),
        match_format=match_format,
        final_set_tiebreak=final_set_tiebreak,
    )


def _win_game(state, config, winner, n_points=4):
    """Win a game by winning n_points in a row (love game)."""
    for _ in range(n_points):
        state = ScoringEngine.point_won_by(state, config, winner)
    return state


def _win_set(state, config, winner, games=6):
    """Win a set 6-0 by winning 6 love games."""
    for _ in range(games):
        state = _win_game(state, config, winner)
    return state


# --- Regular game tests ---


class TestRegularGame:
    def test_love_game_server(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)
        assert state.current_game.server == ServerSide.PLAYER_A

        # Point 1: 15-0
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_game.point_state.server_points == "15"
        assert state.current_game.point_state.returner_points == "0"

        # Point 2: 30-0
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_game.point_state.server_points == "30"

        # Point 3: 40-0
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_game.point_state.server_points == "40"

        # Point 4: game won — 1-0
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_set.games_a == 1
        assert state.current_set.games_b == 0
        # Server should switch
        assert state.current_game.server == ServerSide.PLAYER_B

    def test_love_game_returner(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # Returner (B) breaks serve: wins 4 points
        for _ in range(4):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)

        assert state.current_set.games_a == 0
        assert state.current_set.games_b == 1


class TestDeuceAdvantage:
    def test_deuce_to_advantage_to_game(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # Get to 40-40 (deuce)
        for _ in range(3):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        for _ in range(3):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)

        assert state.current_game.point_state.server_points == "40"
        assert state.current_game.point_state.returner_points == "40"

        # Server gets advantage
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_game.point_state.server_points == "AD"
        assert state.current_game.point_state.returner_points == "40"

        # Server wins game
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_set.games_a == 1

    def test_deuce_back_to_deuce(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # Get to 40-40
        for _ in range(3):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        for _ in range(3):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)

        # AD-40
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state.current_game.point_state.server_points == "AD"

        # Back to deuce (returner wins)
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)
        assert state.current_game.point_state.server_points == "40"
        assert state.current_game.point_state.returner_points == "40"

    def test_returner_advantage_and_break(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # Get to 40-40
        for _ in range(3):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        for _ in range(3):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)

        # Returner gets advantage
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)
        assert state.current_game.point_state.returner_points == "AD"

        # Returner wins (break!)
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)
        assert state.current_set.games_b == 1
        assert state.current_set.games_a == 0


class TestTiebreak:
    def _get_to_tiebreak(self, config):
        """Play to 6-6 and enter tiebreak."""
        state = ScoringEngine.start_match(config)
        # Alternate holds: each player wins their service games to get 6-6
        for i in range(12):
            server = state.current_game.server
            state = _win_game(state, config, server)
        assert state.current_set.games_a == 6
        assert state.current_set.games_b == 6
        assert state.current_game.set_type == SetType.TIEBREAK
        return state

    def test_tiebreak_triggers_at_6_6(self):
        config = _make_config()
        state = self._get_to_tiebreak(config)
        assert state.current_set.is_tiebreak is True

    def test_tiebreak_7_0(self):
        config = _make_config()
        state = self._get_to_tiebreak(config)

        # Player A wins 7 straight tiebreak points
        for _ in range(7):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)

        # Set should be complete: 7-6
        assert len(state.completed_sets) >= 1
        last_set = state.completed_sets[-1]
        assert last_set.games_a == 7
        assert last_set.games_b == 6
        assert last_set.winner == ServerSide.PLAYER_A
        assert last_set.is_tiebreak is True

    def test_tiebreak_must_win_by_2(self):
        config = _make_config()
        state = self._get_to_tiebreak(config)

        # Get to 6-6 in tiebreak
        for _ in range(6):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_B)

        # Still in tiebreak
        assert state.current_set is not None
        assert state.current_set.tiebreak_points_a == 6
        assert state.current_set.tiebreak_points_b == 6

        # A wins 2 more
        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        # 7-6 but need win by 2
        assert state.current_set is not None  # still going

        state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        # 8-6 — set won
        last_set = state.completed_sets[-1]
        assert last_set.winner == ServerSide.PLAYER_A


class TestSetCompletion:
    def test_set_6_0(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # A wins 6 love games
        for _ in range(6):
            state = _win_game(state, config, ServerSide.PLAYER_A)

        assert state.sets_a == 1
        assert state.sets_b == 0
        assert len(state.completed_sets) == 1
        assert state.completed_sets[0].games_a == 6
        assert state.completed_sets[0].games_b == 0

    def test_set_7_5(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # Play to 5-5 (alternate holds)
        for _ in range(10):
            server = state.current_game.server
            state = _win_game(state, config, server)

        assert state.current_set.games_a == 5
        assert state.current_set.games_b == 5

        # A holds: 6-5
        state = _win_game(state, config, state.current_game.server)
        # B holds: 6-6 → would trigger tiebreak, so let's do 7-5 instead
        # Actually at 5-5, A serves, holds to 6-5, B serves, A breaks for 7-5
        # Need A to break B's serve at 6-5
        state = _win_game(state, config, ServerSide.PLAYER_A)  # A breaks
        assert state.sets_a == 1
        last_set = state.completed_sets[-1]
        assert last_set.games_a == 7
        assert last_set.games_b == 5


class TestMatchCompletion:
    def test_straight_sets_best_of_3(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # A wins two sets 6-0, 6-0
        for _ in range(2):
            for _ in range(6):
                state = _win_game(state, config, ServerSide.PLAYER_A)

        assert state.status == MatchStatus.COMPLETED
        assert state.sets_a == 2
        assert state.sets_b == 0
        assert state.is_complete

    def test_three_sets_best_of_3(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # A wins first set 6-0
        for _ in range(6):
            state = _win_game(state, config, ServerSide.PLAYER_A)
        assert state.sets_a == 1

        # B wins second set 6-0
        for _ in range(6):
            state = _win_game(state, config, ServerSide.PLAYER_B)
        assert state.sets_b == 1

        # A wins third set 6-0
        for _ in range(6):
            state = _win_game(state, config, ServerSide.PLAYER_A)

        assert state.status == MatchStatus.COMPLETED
        assert state.sets_a == 2
        assert state.sets_b == 1

    def test_completed_match_ignores_further_points(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)

        # Win match
        for _ in range(2):
            for _ in range(6):
                state = _win_game(state, config, ServerSide.PLAYER_A)

        assert state.is_complete
        version_before = state.version

        # Try to play more — should be no-op
        state2 = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        assert state2.version == version_before


class TestScoreReconstruction:
    def test_from_completed_scores(self):
        config = _make_config()
        state = ScoringEngine.set_state_from_scores(
            config, [(6, 4), (3, 6), (7, 5)]
        )
        assert state.status == MatchStatus.COMPLETED
        assert state.sets_a == 2
        assert state.sets_b == 1
        assert len(state.completed_sets) == 3

    def test_from_in_progress_scores(self):
        config = _make_config()
        state = ScoringEngine.set_state_from_scores(
            config, [(6, 4), (3, 3)], current_game_score=("30", "15")
        )
        assert state.status == MatchStatus.LIVE
        assert state.sets_a == 1
        assert state.sets_b == 0
        assert state.current_set.games_a == 3
        assert state.current_set.games_b == 3
        assert state.current_game.point_state.server_points == "30"
        assert state.current_game.point_state.returner_points == "15"

    def test_from_tiebreak_score(self):
        config = _make_config()
        state = ScoringEngine.set_state_from_scores(config, [(7, 6), (4, 6)])
        assert state.sets_a == 1
        assert state.sets_b == 1
        assert state.completed_sets[0].is_tiebreak is True

    def test_score_summary(self):
        config = _make_config()
        state = ScoringEngine.set_state_from_scores(
            config, [(6, 4), (3, 6), (7, 5)]
        )
        assert state.score_summary == "6-4 3-6 7-5"


class TestValidation:
    def test_valid_initial_state(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)
        errors = validate_match_state(state, config)
        assert errors == []

    def test_valid_after_points(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)
        for _ in range(10):
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
        errors = validate_match_state(state, config)
        assert errors == []

    def test_valid_completed_match(self):
        config = _make_config()
        state = ScoringEngine.start_match(config)
        for _ in range(2):
            for _ in range(6):
                state = _win_game(state, config, ServerSide.PLAYER_A)
        errors = validate_match_state(state, config)
        assert errors == []


class TestFullMatchSimulation:
    def test_play_entire_match_all_holds_with_tiebreak_winner(self):
        """Server holds in regular games → tiebreak, then A wins tiebreak.

        Pure "server always holds" in a tiebreak produces an infinite
        loop due to symmetric rotation, so in tiebreaks Player A wins
        every point to force completion.
        """
        config = _make_config()
        state = ScoringEngine.start_match(config)
        max_points = 1000  # safety limit

        points = 0
        while not state.is_complete and points < max_points:
            if state.current_game.set_type == SetType.TIEBREAK:
                # Break symmetry: A wins every tiebreak point
                winner = ServerSide.PLAYER_A
            else:
                winner = state.current_game.server
            state = ScoringEngine.point_won_by(state, config, winner)
            points += 1
            errors = validate_match_state(state, config)
            assert errors == [], f"Invalid state after point {points}: {errors}"

        assert state.is_complete
        assert points < max_points, "Match did not complete within limit"
        assert state.sets_a == 2 or state.sets_b == 2

    def test_play_entire_match_player_a_dominates(self):
        """Player A wins every point — straight sets 6-0 6-0."""
        config = _make_config()
        state = ScoringEngine.start_match(config)

        points = 0
        while not state.is_complete and points < 500:
            state = ScoringEngine.point_won_by(state, config, ServerSide.PLAYER_A)
            points += 1

        assert state.is_complete
        assert state.sets_a == 2
        assert state.sets_b == 0
