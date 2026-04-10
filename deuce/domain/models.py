"""Pydantic domain models for tennis match state.

These are the canonical data structures for the entire system.
All provider data, scoring engine output, storage, and API responses
are expressed in terms of these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from deuce.domain.enums import (
    MatchFormat,
    MatchStatus,
    ServerSide,
    SetType,
    Surface,
)


class Player(BaseModel):
    model_config = ConfigDict(frozen=True)

    player_id: str
    name: str
    country_code: str = ""
    rank: Optional[int] = None
    seed: Optional[int] = None


class Tournament(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    surface: Surface = Surface.HARD
    tour: str = ""  # "ATP", "WTA"
    level: str = ""  # "250", "500", "1000", "Grand Slam", "Challenger"
    location: str = ""
    year: int = 0


class MatchConfig(BaseModel):
    """Immutable match configuration set at creation time."""

    model_config = ConfigDict(frozen=True)

    match_id: str
    tournament: Tournament
    player_a: Player
    player_b: Player
    match_format: MatchFormat = MatchFormat.BEST_OF_3
    final_set_tiebreak: bool = True
    round_name: str = ""
    court: str = ""
    scheduled_start: Optional[datetime] = None

    @property
    def sets_to_win(self) -> int:
        return 3 if self.match_format == MatchFormat.BEST_OF_5 else 2


class PointState(BaseModel):
    """Current point score within a game."""

    server_points: str = "0"
    returner_points: str = "0"
    is_break_point: bool = False
    is_set_point: bool = False
    is_match_point: bool = False


class GameState(BaseModel):
    """State of the current game within a set."""

    game_number: int = 1
    server: ServerSide = ServerSide.PLAYER_A
    point_state: PointState = Field(default_factory=PointState)
    set_type: SetType = SetType.REGULAR
    points_played: int = 0


class SetState(BaseModel):
    """State of a single set."""

    set_number: int
    games_a: int = 0
    games_b: int = 0
    tiebreak_points_a: int = 0
    tiebreak_points_b: int = 0
    is_tiebreak: bool = False
    is_complete: bool = False
    winner: Optional[ServerSide] = None

    @property
    def score_display(self) -> str:
        s = f"{self.games_a}-{self.games_b}"
        if self.is_tiebreak and self.is_complete:
            tb_loser = min(self.tiebreak_points_a, self.tiebreak_points_b)
            s += f"({tb_loser})"
        return s


class MatchState(BaseModel):
    """Full mutable match state — the canonical internal representation."""

    match_id: str
    status: MatchStatus = MatchStatus.SCHEDULED
    sets_a: int = 0
    sets_b: int = 0
    current_set: Optional[SetState] = None
    completed_sets: list[SetState] = Field(default_factory=list)
    current_game: Optional[GameState] = None
    server: ServerSide = ServerSide.PLAYER_A
    points_played_total: int = 0
    last_updated: Optional[datetime] = None
    version: int = 0

    @property
    def is_complete(self) -> bool:
        return self.status in (
            MatchStatus.COMPLETED,
            MatchStatus.RETIRED,
            MatchStatus.WALKOVER,
        )

    @property
    def score_summary(self) -> str:
        parts = [s.score_display for s in self.completed_sets]
        if self.current_set and not self.current_set.is_complete:
            cs = self.current_set
            parts.append(f"{cs.games_a}-{cs.games_b}*")
        return " ".join(parts) if parts else "0-0"
