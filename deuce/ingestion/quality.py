"""Data-quality checks for historical match loads.

Error-level failures are schema gaps, required nulls, duplicate identities,
unusable dates, out-of-range values, and referential contradictions. Unknown
optional codes are warnings. The import job decides whether errors block the
load or only the bad rows are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from deuce.ingestion.adapters import MatchSourceRecord
from deuce.name_utils import normalize_player_name

REQUIRED_COLUMNS = (
    "player_a",
    "player_b",
    "winner",
    "date",
    "tournament",
    "surface",
)
SURFACES = {"Hard", "Clay", "Grass", "Carpet"}
KNOWN_LEVELS = {"G", "M", "500", "250", "F", "A", "O", "D"}
WINNERS = {"A", "B"}
RANK_MIN = 1
RANK_MAX = 5000
POINTS_MIN = 0.0
POINTS_MAX = 100_000.0
BEST_OF_VALUES = {3, 5}


@dataclass(frozen=True)
class DQIssue:
    check: str
    severity: str
    message: str
    row_index: int | None = None
    field: str | None = None


@dataclass(frozen=True)
class NormalizedMatch:
    match_key: str
    player_a: str
    player_b: str
    winner: str
    match_date: str
    tournament: str
    surface: str
    round_name: str
    rank_a: int | None
    rank_b: int | None
    points_a: float | None
    points_b: float | None
    best_of: int | None
    tourney_level: str
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class DQReport:
    issues: list[DQIssue]
    valid: list[NormalizedMatch]
    rows_seen: int

    @property
    def errors(self) -> list[DQIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_match_date(value: str) -> date | None:
    text = (value or "").strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def canonical_date_token(value: str) -> str:
    parsed = parse_match_date(value)
    if parsed is None:
        return (value or "").strip()
    return parsed.strftime("%Y%m%d")


def schema_issues(fieldnames: list[str] | None) -> list[DQIssue]:
    present = set(fieldnames or [])
    missing = [column for column in REQUIRED_COLUMNS if column not in present]
    if not missing:
        return []
    return [
        DQIssue(
            check="schema",
            severity="error",
            message="missing required columns: " + ", ".join(missing),
            field=",".join(missing),
        )
    ]


def evaluate_records(
    records: list[MatchSourceRecord],
    *,
    date_min: date,
    date_max: date,
) -> DQReport:
    issues: list[DQIssue] = []
    valid: list[NormalizedMatch] = []
    seen: dict[str, int] = {}

    for index, record in enumerate(records):
        row_issues, normalized = _check_row(
            record,
            index,
            date_min=date_min,
            date_max=date_max,
        )
        if record.match_key in seen:
            row_issues.append(
                DQIssue(
                    check="duplicates",
                    severity="error",
                    message=f"duplicate match_key {record.match_key} (first row {seen[record.match_key]})",
                    row_index=index,
                    field="match_key",
                )
            )
        else:
            seen[record.match_key] = index
        issues.extend(row_issues)
        if normalized is not None and not any(issue.severity == "error" for issue in row_issues):
            valid.append(normalized)

    return DQReport(issues=issues, valid=valid, rows_seen=len(records))


def _check_row(
    record: MatchSourceRecord,
    index: int,
    *,
    date_min: date,
    date_max: date,
) -> tuple[list[DQIssue], NormalizedMatch | None]:
    issues: list[DQIssue] = []
    required = {
        "player_a": record.player_a,
        "player_b": record.player_b,
        "winner": record.winner,
        "date": record.date,
        "tournament": record.tournament,
        "surface": record.surface,
    }
    for field_name, value in required.items():
        if not (value or "").strip():
            issues.append(
                DQIssue(
                    check="nulls",
                    severity="error",
                    message=f"{field_name} is empty",
                    row_index=index,
                    field=field_name,
                )
            )

    parsed_date = parse_match_date(record.date)
    if (record.date or "").strip() and parsed_date is None:
        issues.append(
            DQIssue(
                check="date_range",
                severity="error",
                message=f"unparseable date {record.date!r}",
                row_index=index,
                field="date",
            )
        )
    elif parsed_date is not None and (parsed_date < date_min or parsed_date > date_max):
        issues.append(
            DQIssue(
                check="date_range",
                severity="error",
                message=f"date {parsed_date.isoformat()} outside {date_min.isoformat()}..{date_max.isoformat()}",
                row_index=index,
                field="date",
            )
        )

    surface = (record.surface or "").strip().title()
    if (record.surface or "").strip() and surface not in SURFACES:
        issues.append(
            DQIssue(
                check="value_range",
                severity="error",
                message=f"surface {record.surface!r} is not one of {sorted(SURFACES)}",
                row_index=index,
                field="surface",
            )
        )

    winner = (record.winner or "").strip().upper()
    if (record.winner or "").strip() and winner not in WINNERS:
        issues.append(
            DQIssue(
                check="value_range",
                severity="error",
                message="winner must be A or B",
                row_index=index,
                field="winner",
            )
        )

    rank_a, rank_a_issue = _optional_int(record.rank_a, field="rank_a", low=RANK_MIN, high=RANK_MAX)
    rank_b, rank_b_issue = _optional_int(record.rank_b, field="rank_b", low=RANK_MIN, high=RANK_MAX)
    points_a, points_a_issue = _optional_float(record.points_a, field="points_a", low=POINTS_MIN, high=POINTS_MAX)
    points_b, points_b_issue = _optional_float(record.points_b, field="points_b", low=POINTS_MIN, high=POINTS_MAX)
    best_of, best_of_issue = _optional_choice(record.best_of, field="best_of", allowed=BEST_OF_VALUES)
    for issue in (rank_a_issue, rank_b_issue, points_a_issue, points_b_issue, best_of_issue):
        if issue is not None:
            issues.append(
                DQIssue(
                    check="value_range",
                    severity="error",
                    message=issue[1],
                    row_index=index,
                    field=issue[0],
                )
            )

    level = (record.tourney_level or "").strip()
    if level and level not in KNOWN_LEVELS:
        issues.append(
            DQIssue(
                check="value_range",
                severity="warning",
                message=f"unknown tourney_level {level!r}",
                row_index=index,
                field="tourney_level",
            )
        )

    player_a = (record.player_a or "").strip()
    player_b = (record.player_b or "").strip()
    if player_a and player_b and normalize_player_name(player_a) == normalize_player_name(player_b):
        issues.append(
            DQIssue(
                check="referential",
                severity="error",
                message="player_a and player_b are the same player",
                row_index=index,
                field="player_b",
            )
        )
    if winner in WINNERS:
        selected = player_a if winner == "A" else player_b
        if not selected.strip():
            issues.append(
                DQIssue(
                    check="referential",
                    severity="error",
                    message=f"winner {winner} does not resolve to a player name",
                    row_index=index,
                    field="winner",
                )
            )

    if any(issue.severity == "error" for issue in issues):
        return issues, None
    if parsed_date is None:
        return issues, None

    return issues, NormalizedMatch(
        match_key=record.match_key,
        player_a=player_a,
        player_b=player_b,
        winner=winner,
        match_date=parsed_date.strftime("%Y%m%d"),
        tournament=(record.tournament or "").strip(),
        surface=surface,
        round_name=(record.round or "").strip(),
        rank_a=rank_a,
        rank_b=rank_b,
        points_a=points_a,
        points_b=points_b,
        best_of=best_of,
        tourney_level=level,
        raw=dict(record.raw),
    )


def _optional_int(value: str, *, field: str, low: int, high: int) -> tuple[int | None, tuple[str, str] | None]:
    text = (value or "").strip()
    if not text:
        return None, None
    try:
        number = float(text)
    except ValueError:
        return None, (field, f"{field} is not a number")
    if not number.is_integer():
        return None, (field, f"{field} is not an integer")
    parsed = int(number)
    if parsed < low or parsed > high:
        return None, (field, f"{field} {parsed} outside {low}..{high}")
    return parsed, None


def _optional_float(
    value: str,
    *,
    field: str,
    low: float,
    high: float,
) -> tuple[float | None, tuple[str, str] | None]:
    text = (value or "").strip()
    if not text:
        return None, None
    try:
        parsed = float(text)
    except ValueError:
        return None, (field, f"{field} is not a number")
    if parsed < low or parsed > high:
        return None, (field, f"{field} {parsed} outside {low}..{high}")
    return parsed, None


def _optional_choice(
    value: str,
    *,
    field: str,
    allowed: set[int],
) -> tuple[int | None, tuple[str, str] | None]:
    text = (value or "").strip()
    if not text:
        return None, None
    try:
        number = float(text)
    except ValueError:
        return None, (field, f"{field} is not a number")
    if not number.is_integer():
        return None, (field, f"{field} is not an integer")
    parsed = int(number)
    if parsed not in allowed:
        return None, (field, f"{field} must be one of {sorted(allowed)}")
    return parsed, None


def issue_to_dict(issue: DQIssue) -> dict[str, Any]:
    return {
        "check": issue.check,
        "severity": issue.severity,
        "message": issue.message,
        "row_index": issue.row_index,
        "field": issue.field,
    }
