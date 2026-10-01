"""Create append-only claim event history.

Revision ID: 20261001_0004
Revises: 20261001_0003
Create Date: 2026-10-01
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "20261001_0004"
down_revision: str | None = "20261001_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "claim_events",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("assessment_session_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column("field_path", sa.Text(), nullable=True),
        sa.Column("old_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("raw_answer", sa.Text(), nullable=True),
        sa.Column("normalized_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("missing_information_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "evidence_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("related_phase", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "event_type IN ('fnol_submitted', 'claim_fact_extracted', "
            "'claim_fact_updated', 'missing_information_created', "
            "'user_answer_received', 'user_answer_applied', 'provider_error', "
            "'phase_completed', 'selective_rerun_started', "
            "'selective_rerun_completed', 'human_handoff_created')",
            name="ck_claim_events_event_type",
        ),
        sa.CheckConstraint(
            "actor_type IN ('user', 'agent', 'system', 'provider', 'human_reviewer')",
            name="ck_claim_events_actor_type",
        ),
        sa.CheckConstraint(
            "related_phase IS NULL OR related_phase IN ("
            "'claim_facts_extraction', 'fact_validation', 'coverage_assessment', "
            "'exclusion_assessment', 'obligation_assessment', 'claim_calculation', "
            "'claim_recommendation')",
            name="ck_claim_events_related_phase",
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claim_agent_states.claim_id"],
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ix_claim_events_claim_created",
        "claim_events",
        ["claim_id", "created_at", "event_id"],
    )
    op.create_index(
        "ix_claim_events_claim_field_created",
        "claim_events",
        ["claim_id", "field_path", "created_at"],
    )
    op.execute("""
        CREATE FUNCTION reject_claim_event_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'claim_events is append-only';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER trg_claim_events_append_only
        BEFORE UPDATE OR DELETE ON claim_events
        FOR EACH ROW EXECUTE FUNCTION reject_claim_event_mutation()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_claim_events_append_only ON claim_events")
    op.execute("DROP FUNCTION IF EXISTS reject_claim_event_mutation()")
    op.drop_index("ix_claim_events_claim_field_created", table_name="claim_events")
    op.drop_index("ix_claim_events_claim_created", table_name="claim_events")
    op.drop_table("claim_events")
