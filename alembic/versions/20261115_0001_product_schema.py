"""Product schema for live storage and historical imports.

Revision ID: 20261115_0001
Revises:
Create Date: 2026-11-15
"""

from alembic import op
import sqlalchemy as sa

revision = "20261115_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_key", sa.String(length=128), nullable=False),
        sa.Column("date_str", sa.String(length=16), nullable=False, server_default=sa.text("''")),
        sa.Column("tournament", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("draw", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("source", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("status_text", sa.String(length=256), nullable=False, server_default=sa.text("''")),
        sa.Column("is_live", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_scheduled", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("raw_row_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("domain_state_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("match_key", name="uq_matches_match_key"),
    )
    op.create_index("ix_matches_date_str", "matches", ["date_str"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("match_id", "sequence", name="uq_events_match_sequence"),
    )
    op.create_index("ix_events_match_id", "events", ["match_id"])
    op.create_index("ix_events_sequence", "events", ["sequence"])

    op.create_table(
        "snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("provider_id", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("payload_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_snapshots_match_id", "snapshots", ["match_id"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id", ondelete="CASCADE"), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False, server_default=sa.text("'default'")),
        sa.Column("prob_a", sa.Float(), nullable=True),
        sa.Column("prob_b", sa.Float(), nullable=True),
        sa.Column("extras_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])

    op.create_table(
        "raw_payloads",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("payload_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_raw_payloads_source", "raw_payloads", ["source"])

    op.create_table(
        "import_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("source_id", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("''")),
        sa.Column("mode", sa.String(length=16), nullable=False, server_default=sa.text("'block'")),
        sa.Column("rows_seen", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_loaded", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rows_rejected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("issues_json", sa.Text(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("message", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_import_runs_source_id", "import_runs", ["source_id"])
    op.create_index("ix_import_runs_status", "import_runs", ["status"])

    op.create_table(
        "historical_matches",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("match_key", sa.String(length=64), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False, server_default=sa.text("''")),
        sa.Column("import_run_id", sa.Integer(), sa.ForeignKey("import_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("match_date", sa.String(length=16), nullable=False, server_default=sa.text("''")),
        sa.Column("tournament", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("round_name", sa.String(length=32), nullable=False, server_default=sa.text("''")),
        sa.Column("surface", sa.String(length=16), nullable=False, server_default=sa.text("''")),
        sa.Column("player_a", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("player_b", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("winner", sa.String(length=8), nullable=False, server_default=sa.text("''")),
        sa.Column("rank_a", sa.Integer(), nullable=True),
        sa.Column("rank_b", sa.Integer(), nullable=True),
        sa.Column("points_a", sa.Float(), nullable=True),
        sa.Column("points_b", sa.Float(), nullable=True),
        sa.Column("best_of", sa.Integer(), nullable=True),
        sa.Column("tourney_level", sa.String(length=16), nullable=False, server_default=sa.text("''")),
        sa.Column("payload_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("match_key", name="uq_historical_matches_match_key"),
    )
    op.create_index("ix_historical_matches_source_id", "historical_matches", ["source_id"])
    op.create_index("ix_historical_matches_import_run_id", "historical_matches", ["import_run_id"])
    op.create_index("ix_historical_matches_match_date", "historical_matches", ["match_date"])


def downgrade() -> None:
    op.drop_table("historical_matches")
    op.drop_table("import_runs")
    op.drop_table("raw_payloads")
    op.drop_table("predictions")
    op.drop_table("snapshots")
    op.drop_table("events")
    op.drop_table("matches")
