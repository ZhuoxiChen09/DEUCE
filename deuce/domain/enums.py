"""Domain enumerations for tennis match state."""

from enum import Enum


class MatchFormat(str, Enum):
    BEST_OF_3 = "best_of_3"
    BEST_OF_5 = "best_of_5"


class Surface(str, Enum):
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"


class MatchStatus(str, Enum):
    SCHEDULED = "scheduled"
    WARMUP = "warmup"
    LIVE = "live"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    RETIRED = "retired"
    WALKOVER = "walkover"
    CANCELLED = "cancelled"


class PointScore(str, Enum):
    ZERO = "0"
    FIFTEEN = "15"
    THIRTY = "30"
    FORTY = "40"
    ADVANTAGE = "AD"


class SetType(str, Enum):
    REGULAR = "regular"
    TIEBREAK = "tiebreak"
    SUPER_TIEBREAK = "super_tiebreak"


class ServerSide(str, Enum):
    PLAYER_A = "player_a"
    PLAYER_B = "player_b"
