"""Data-source adapters.

A new dataset is a new :class:`DataSourceAdapter` registered with
:func:`register_adapter`. This module ships one adapter, ``local_csv``, which
reads a CSV already on disk in the DEUCE training schema. It does not fetch
or vendor a new public dataset.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from deuce.config.settings import import_adapter_name, import_source_path

AdapterFactory = Callable[..., "DataSourceAdapter"]
_REGISTRY: dict[str, AdapterFactory] = {}


class SourceLoadError(Exception):
    """The adapter could not read its source."""


@dataclass(frozen=True)
class MatchSourceRecord:
    match_key: str
    player_a: str
    player_b: str
    winner: str
    date: str
    tournament: str
    surface: str
    round: str = ""
    rank_a: str = ""
    rank_b: str = ""
    points_a: str = ""
    points_b: str = ""
    best_of: str = ""
    tourney_level: str = ""
    raw: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadResult:
    source_id: str
    fieldnames: list[str]
    records: list[MatchSourceRecord]


class DataSourceAdapter:
    """Read-only source of historical match rows.

    Subclasses implement :meth:`load`. Register the subclass factory under a
    stable name so import configuration can select it without editing the job.
    """

    @property
    def source_id(self) -> str:
        raise NotImplementedError

    def load(self) -> LoadResult:
        raise NotImplementedError


def build_match_key(
    source_id: str,
    date_value: str,
    tournament: str,
    round_name: str,
    player_a: str,
    player_b: str,
) -> str:
    from deuce.ingestion.quality import canonical_date_token

    raw = "|".join(
        [
            source_id.strip(),
            canonical_date_token(date_value),
            tournament.strip(),
            round_name.strip(),
            player_a.strip(),
            player_b.strip(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def register_adapter(name: str, factory: AdapterFactory) -> None:
    key = name.strip()
    if not key:
        raise ValueError("adapter name is required")
    _REGISTRY[key] = factory


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_adapter(name: str | None = None, *, path: str | Path | None = None) -> DataSourceAdapter:
    key = (name or import_adapter_name()).strip()
    try:
        factory = _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(registered_adapters()) or "(none)"
        raise KeyError(f"Unknown data-source adapter {key!r}. Known adapters: {known}") from exc
    if path is None:
        return factory()
    return factory(path=path)


class LocalCsvMatchAdapter(DataSourceAdapter):
    """CSV already stored in the repository or supplied via ``DEUCE_IMPORT_SOURCE``."""

    source_name = "local_csv"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else import_source_path()

    @property
    def source_id(self) -> str:
        return self.source_name

    def load(self) -> LoadResult:
        if not self.path.exists():
            raise SourceLoadError(f"source file not found: {self.path}")
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            records: list[MatchSourceRecord] = []
            for row in reader:
                records.append(self._record(row))
        return LoadResult(source_id=self.source_id, fieldnames=fieldnames, records=records)

    def _record(self, row: dict[str, str | None]) -> MatchSourceRecord:
        cleaned = {str(key): "" if value is None else str(value) for key, value in row.items() if key is not None}

        def cell(name: str) -> str:
            return cleaned.get(name, "").strip()

        player_a = cell("player_a")
        player_b = cell("player_b")
        date_value = cell("date")
        tournament = cell("tournament")
        round_name = cell("round")
        return MatchSourceRecord(
            match_key=build_match_key(self.source_id, date_value, tournament, round_name, player_a, player_b),
            player_a=player_a,
            player_b=player_b,
            winner=cell("winner"),
            date=date_value,
            tournament=tournament,
            surface=cell("surface"),
            round=round_name,
            rank_a=cell("rank_a"),
            rank_b=cell("rank_b"),
            points_a=cell("points_a"),
            points_b=cell("points_b"),
            best_of=cell("best_of"),
            tourney_level=cell("tourney_level"),
            raw=cleaned,
        )


def _local_csv_factory(path: str | Path | None = None) -> LocalCsvMatchAdapter:
    return LocalCsvMatchAdapter(path)


register_adapter(LocalCsvMatchAdapter.source_name, _local_csv_factory)
