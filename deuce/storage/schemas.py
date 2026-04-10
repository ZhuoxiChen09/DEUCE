"""SQLAlchemy ORM models for live score persistence (SQLite v1)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MatchRow(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    date_str: Mapped[str] = mapped_column(String(16), index=True, default="")
    tournament: Mapped[str] = mapped_column(Text, default="")
    draw: Mapped[str] = mapped_column(String(64), default="")
    source: Mapped[str] = mapped_column(String(64), default="")
    status_text: Mapped[str] = mapped_column(String(256), default="")
    is_live: Mapped[int] = mapped_column(Integer, default=0)
    is_completed: Mapped[int] = mapped_column(Integer, default=0)
    is_scheduled: Mapped[int] = mapped_column(Integer, default=0)
    raw_row_json: Mapped[str] = mapped_column(Text, default="{}")
    domain_state_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    events: Mapped[list["EventRow"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    snapshots: Mapped[list["SnapshotRow"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    predictions: Mapped[list["PredictionRow"]] = relationship(back_populates="match", cascade="all, delete-orphan")


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    sequence: Mapped[int] = mapped_column(Integer, index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match: Mapped["MatchRow"] = relationship(back_populates="events")

    __table_args__ = (UniqueConstraint("match_id", "sequence", name="uq_events_match_sequence"),)


class SnapshotRow(Base):
    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    provider_id: Mapped[str] = mapped_column(String(64), default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match: Mapped["MatchRow"] = relationship(back_populates="snapshots")


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[str] = mapped_column(String(64), default="default")
    prob_a: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_b: Mapped[float | None] = mapped_column(Float, nullable=True)
    extras_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    match: Mapped["MatchRow"] = relationship(back_populates="predictions")


class RawPayloadRow(Base):
    __tablename__ = "raw_payloads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), index=True, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
