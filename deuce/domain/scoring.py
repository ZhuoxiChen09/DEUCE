"""Tennis scoring engine — pure-function state machine.

All methods are static. State transitions return new model instances.
No mutation, no side effects, no I/O.

Tennis scoring rules implemented:
- Regular game: 0 → 15 → 30 → 40 → game. At 40-40 (deuce): AD → game or back to deuce.
- Tiebreak at 6-6: first to 7 points, win by 2. Server alternates every 2 points after first.
- Super tiebreak (10-point) in final set when configured.
- Set won at 6 games with 2-game lead, or 7-6 via tiebreak.
- Match won when a player reaches sets_to_win (2 for best-of-3, 3 for best-of-5).
"""

from __future__ import annotations

from datetime import datetime, timezone

from deuce.domain.enums import (
    MatchFormat,
    MatchStatus,
    ServerSide,
    SetType,
)
from deuce.domain.models import (
    GameState,
    MatchConfig,
    MatchState,
    PointState,
    SetState,
)

_NEXT_POINT = {"0": "15", "15": "30", "30": "40"}


def _opposite(side: ServerSide) -> ServerSide:
    if side == ServerSide.PLAYER_A:
        return ServerSide.PLAYER_B
    return ServerSide.PLAYER_A


class ScoringEngine:
    """Pure-function tennis scoring state machine."""

    @staticmethod
    def start_match(config: MatchConfig) -> MatchState:
        first_set = SetState(set_number=1)
        first_game = GameState(
            game_number=1,
            server=ServerSide.PLAYER_A,
            point_state=PointState(),
            set_type=SetType.REGULAR,
        )
        return MatchState(
            match_id=config.match_id,
            status=MatchStatus.LIVE,
            current_set=first_set,
            current_game=first_game,
            server=ServerSide.PLAYER_A,
            version=1,
            last_updated=datetime.now(timezone.utc),
        )

    @staticmethod
    def point_won_by(
        state: MatchState,
        config: MatchConfig,
        winner: ServerSide,
    ) -> MatchState:
        """Apply a single point won by *winner*. Returns new MatchState."""
        if state.is_complete:
            return state

        assert state.current_set is not None
        assert state.current_game is not None

        cur_set = state.current_set
        cur_game = state.current_game
        new_version = state.version + 1
        now = datetime.now(timezone.utc)

        # --- Tiebreak point ---
        if cur_game.set_type in (SetType.TIEBREAK, SetType.SUPER_TIEBREAK):
            target = 10 if cur_game.set_type == SetType.SUPER_TIEBREAK else 7
            new_set, tb_won, tb_winner = ScoringEngine._advance_tiebreak(
                cur_set, winner, target=target
            )
            if tb_won:
                return ScoringEngine._complete_set(
                    state, config, new_set, tb_winner, new_version, now
                )
            # Tiebreak continues — update point count and server rotation
            total_tb_pts = new_set.tiebreak_points_a + new_set.tiebreak_points_b
            # Server changes after first point, then every 2 points
            if total_tb_pts == 1 or (total_tb_pts > 1 and (total_tb_pts - 1) % 2 == 0):
                new_server = _opposite(cur_game.server)
            else:
                new_server = cur_game.server
            new_game = GameState(
                game_number=cur_game.game_number,
                server=new_server,
                point_state=PointState(),
                set_type=cur_game.set_type,
                points_played=cur_game.points_played + 1,
            )
            return MatchState(
                match_id=state.match_id,
                status=MatchStatus.LIVE,
                sets_a=state.sets_a,
                sets_b=state.sets_b,
                current_set=new_set,
                completed_sets=list(state.completed_sets),
                current_game=new_game,
                server=new_server,
                points_played_total=state.points_played_total + 1,
                last_updated=now,
                version=new_version,
            )

        # --- Regular game point ---
        server_won = winner == cur_game.server
        new_pts, game_won, won_by_server = ScoringEngine._advance_regular_game(
            cur_game.point_state, server_won
        )
        game_winner = cur_game.server if won_by_server else _opposite(cur_game.server)

        if not game_won:
            new_game = GameState(
                game_number=cur_game.game_number,
                server=cur_game.server,
                point_state=new_pts,
                set_type=SetType.REGULAR,
                points_played=cur_game.points_played + 1,
            )
            return MatchState(
                match_id=state.match_id,
                status=MatchStatus.LIVE,
                sets_a=state.sets_a,
                sets_b=state.sets_b,
                current_set=cur_set,
                completed_sets=list(state.completed_sets),
                current_game=new_game,
                server=state.server,
                points_played_total=state.points_played_total + 1,
                last_updated=now,
                version=new_version,
            )

        # Game won — advance set score
        if game_winner == ServerSide.PLAYER_A:
            new_set = cur_set.model_copy(update={"games_a": cur_set.games_a + 1})
        else:
            new_set = cur_set.model_copy(update={"games_b": cur_set.games_b + 1})

        # Check if set is won
        set_won, set_winner = ScoringEngine._check_set_won(new_set, config, state)
        if set_won:
            return ScoringEngine._complete_set(
                state, config, new_set, set_winner, new_version, now
            )

        # Check if tiebreak should start
        if new_set.games_a == 6 and new_set.games_b == 6:
            is_final_set = (state.sets_a + state.sets_b + 1) == (
                config.sets_to_win * 2 - 1
            )
            if is_final_set and not config.final_set_tiebreak:
                # Advantage final set — no tiebreak, keep playing regular games
                pass
            else:
                tb_type = SetType.TIEBREAK
                if is_final_set and config.final_set_tiebreak:
                    # Some tours use super tiebreak in final set
                    # Default to regular tiebreak; config can override
                    tb_type = SetType.TIEBREAK
                new_set = new_set.model_copy(update={"is_tiebreak": True})
                # In tiebreak, first server is the player who would have served next
                next_server = _opposite(cur_game.server)
                new_game = GameState(
                    game_number=cur_game.game_number + 1,
                    server=next_server,
                    point_state=PointState(),
                    set_type=tb_type,
                    points_played=0,
                )
                return MatchState(
                    match_id=state.match_id,
                    status=MatchStatus.LIVE,
                    sets_a=state.sets_a,
                    sets_b=state.sets_b,
                    current_set=new_set,
                    completed_sets=list(state.completed_sets),
                    current_game=new_game,
                    server=next_server,
                    points_played_total=state.points_played_total + 1,
                    last_updated=now,
                    version=new_version,
                )

        # Normal new game — swap server
        next_server = _opposite(cur_game.server)
        new_game = GameState(
            game_number=cur_game.game_number + 1,
            server=next_server,
            point_state=PointState(),
            set_type=SetType.REGULAR,
            points_played=0,
        )
        return MatchState(
            match_id=state.match_id,
            status=MatchStatus.LIVE,
            sets_a=state.sets_a,
            sets_b=state.sets_b,
            current_set=new_set,
            completed_sets=list(state.completed_sets),
            current_game=new_game,
            server=next_server,
            points_played_total=state.points_played_total + 1,
            last_updated=now,
            version=new_version,
        )

    @staticmethod
    def _advance_regular_game(
        pts: PointState, server_won: bool
    ) -> tuple[PointState, bool, bool]:
        """Returns (new_point_state, game_is_won, won_by_server).

        won_by_server is only meaningful when game_is_won=True.
        """
        sp, rp = pts.server_points, pts.returner_points

        if server_won:
            if sp == "AD":
                return PointState(), True, True
            if sp == "40":
                if rp == "AD":
                    return PointState(server_points="40", returner_points="40"), False, False
                if rp == "40":
                    return PointState(server_points="AD", returner_points="40"), False, False
                return PointState(), True, True
            new_sp = _NEXT_POINT[sp]
            return PointState(server_points=new_sp, returner_points=rp), False, False
        else:
            if rp == "AD":
                return PointState(), True, False
            if rp == "40":
                if sp == "AD":
                    return PointState(server_points="40", returner_points="40"), False, False
                if sp == "40":
                    return PointState(server_points="40", returner_points="AD"), False, False
                return PointState(), True, False
            new_rp = _NEXT_POINT[rp]
            return PointState(server_points=sp, returner_points=new_rp), False, False

    @staticmethod
    def _advance_tiebreak(
        set_state: SetState,
        winner: ServerSide,
        target: int = 7,
    ) -> tuple[SetState, bool, ServerSide | None]:
        """Returns (new_set_state, tiebreak_won, winner_or_None)."""
        if winner == ServerSide.PLAYER_A:
            new_a = set_state.tiebreak_points_a + 1
            new_b = set_state.tiebreak_points_b
        else:
            new_a = set_state.tiebreak_points_a
            new_b = set_state.tiebreak_points_b + 1

        new_set = set_state.model_copy(
            update={"tiebreak_points_a": new_a, "tiebreak_points_b": new_b}
        )

        # Check win condition: >= target AND lead >= 2
        if new_a >= target and new_a - new_b >= 2:
            return new_set, True, ServerSide.PLAYER_A
        if new_b >= target and new_b - new_a >= 2:
            return new_set, True, ServerSide.PLAYER_B
        return new_set, False, None

    @staticmethod
    def _check_set_won(
        set_state: SetState,
        config: MatchConfig,
        match_state: MatchState,
    ) -> tuple[bool, ServerSide | None]:
        """Check if the set is won by normal game scoring (not tiebreak)."""
        ga, gb = set_state.games_a, set_state.games_b
        if ga >= 6 and ga - gb >= 2:
            return True, ServerSide.PLAYER_A
        if gb >= 6 and gb - ga >= 2:
            return True, ServerSide.PLAYER_B
        return False, None

    @staticmethod
    def _complete_set(
        state: MatchState,
        config: MatchConfig,
        set_state: SetState,
        set_winner: ServerSide | None,
        new_version: int,
        now: datetime,
    ) -> MatchState:
        """Finalize a set and start the next one, or complete the match."""
        # If tiebreak, the winner gets 7 games
        final_set = set_state.model_copy(
            update={
                "is_complete": True,
                "winner": set_winner,
                "games_a": 7 if set_winner == ServerSide.PLAYER_A and set_state.is_tiebreak else set_state.games_a,
                "games_b": 7 if set_winner == ServerSide.PLAYER_B and set_state.is_tiebreak else set_state.games_b,
            }
        )
        completed = list(state.completed_sets) + [final_set]
        new_sets_a = state.sets_a + (1 if set_winner == ServerSide.PLAYER_A else 0)
        new_sets_b = state.sets_b + (1 if set_winner == ServerSide.PLAYER_B else 0)

        # Check match won
        if new_sets_a >= config.sets_to_win:
            return MatchState(
                match_id=state.match_id,
                status=MatchStatus.COMPLETED,
                sets_a=new_sets_a,
                sets_b=new_sets_b,
                current_set=None,
                completed_sets=completed,
                current_game=None,
                server=state.server,
                points_played_total=state.points_played_total + 1,
                last_updated=now,
                version=new_version,
            )
        if new_sets_b >= config.sets_to_win:
            return MatchState(
                match_id=state.match_id,
                status=MatchStatus.COMPLETED,
                sets_a=new_sets_a,
                sets_b=new_sets_b,
                current_set=None,
                completed_sets=completed,
                current_game=None,
                server=state.server,
                points_played_total=state.points_played_total + 1,
                last_updated=now,
                version=new_version,
            )

        # Start next set — server after tiebreak returns to the player who served first in it
        # In a normal set end, server continues to alternate
        if state.current_game:
            next_server = _opposite(state.current_game.server)
            # After tiebreak, the player who received first in the tiebreak serves first in new set
            if set_state.is_tiebreak and state.current_game.set_type in (SetType.TIEBREAK, SetType.SUPER_TIEBREAK):
                # The tiebreak server was the receiver from the previous game
                # After tiebreak, the OTHER player serves first
                next_server = _opposite(state.current_game.server)
        else:
            next_server = _opposite(state.server)

        next_set_num = len(completed) + 1
        new_set = SetState(set_number=next_set_num)
        new_game = GameState(
            game_number=1,
            server=next_server,
            point_state=PointState(),
            set_type=SetType.REGULAR,
        )
        return MatchState(
            match_id=state.match_id,
            status=MatchStatus.LIVE,
            sets_a=new_sets_a,
            sets_b=new_sets_b,
            current_set=new_set,
            completed_sets=completed,
            current_game=new_game,
            server=next_server,
            points_played_total=state.points_played_total + 1,
            last_updated=now,
            version=new_version,
        )

    @staticmethod
    def set_state_from_scores(
        config: MatchConfig,
        set_scores: list[tuple[int, int]],
        current_game_score: tuple[str, str] | None = None,
        server: ServerSide = ServerSide.PLAYER_A,
    ) -> MatchState:
        """Reconstruct MatchState from set-level scores.

        Used for reconciling provider data that arrives as set scores
        without point-by-point history.
        """
        completed: list[SetState] = []
        sets_a = 0
        sets_b = 0

        for i, (ga, gb) in enumerate(set_scores):
            set_num = i + 1
            is_tb = (ga == 7 and gb == 6) or (ga == 6 and gb == 7)
            if ga > gb:
                winner = ServerSide.PLAYER_A
                sets_a += 1
            elif gb > ga:
                winner = ServerSide.PLAYER_B
                sets_b += 1
            else:
                # Incomplete set (equal games) — this is the current set
                cur_set = SetState(set_number=set_num, games_a=ga, games_b=gb)
                game = None
                if current_game_score:
                    game = GameState(
                        game_number=ga + gb + 1,
                        server=server,
                        point_state=PointState(
                            server_points=current_game_score[0],
                            returner_points=current_game_score[1],
                        ),
                        set_type=SetType.TIEBREAK if ga == 6 and gb == 6 else SetType.REGULAR,
                    )
                status = MatchStatus.LIVE if game else MatchStatus.LIVE
                return MatchState(
                    match_id=config.match_id,
                    status=status,
                    sets_a=sets_a,
                    sets_b=sets_b,
                    current_set=cur_set,
                    completed_sets=completed,
                    current_game=game,
                    server=server,
                    version=1,
                )

            completed.append(
                SetState(
                    set_number=set_num,
                    games_a=ga,
                    games_b=gb,
                    is_tiebreak=is_tb,
                    is_complete=True,
                    winner=winner,
                )
            )

        # All sets are complete — check if match is done
        is_done = sets_a >= config.sets_to_win or sets_b >= config.sets_to_win
        if is_done:
            return MatchState(
                match_id=config.match_id,
                status=MatchStatus.COMPLETED,
                sets_a=sets_a,
                sets_b=sets_b,
                completed_sets=completed,
                server=server,
                version=1,
            )

        # Match still in progress — start next set
        next_set_num = len(completed) + 1
        cur_set = SetState(set_number=next_set_num)
        game = GameState(
            game_number=1,
            server=server,
            point_state=PointState(),
            set_type=SetType.REGULAR,
        )
        if current_game_score:
            game = game.model_copy(
                update={
                    "point_state": PointState(
                        server_points=current_game_score[0],
                        returner_points=current_game_score[1],
                    )
                }
            )
        return MatchState(
            match_id=config.match_id,
            status=MatchStatus.LIVE,
            sets_a=sets_a,
            sets_b=sets_b,
            current_set=cur_set,
            completed_sets=completed,
            current_game=game,
            server=server,
            version=1,
        )
