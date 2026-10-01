"""Create immutable assessment versions.

Revision ID: 20261001_0005
Revises: 20261001_0004
Create Date: 2026-10-01
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "20261001_0005"
down_revision: str | None = "20261001_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "assessment_versions",
        sa.Column("assessment_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("parent_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("trigger_type", sa.String(length=32), nullable=False),
        sa.Column("trigger_reference_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("rerun_from_phase", sa.String(length=64), nullable=True),
        sa.Column(
            "completed_phases",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("phase_outputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("recommendation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "version_number >= 1", name="ck_assessment_versions_number_positive"
        ),
        sa.CheckConstraint(
            "trigger_type IN ('initial_assessment', 'user_follow_up', "
            "'selective_rerun', 'provider_retry', 'manual_review')",
            name="ck_assessment_versions_trigger_type",
        ),
        sa.CheckConstraint(
            "status IN ('incomplete', 'completed', 'needs_information', "
            "'needs_human_review', 'failed')",
            name="ck_assessment_versions_status",
        ),
        sa.CheckConstraint(
            "rerun_from_phase IS NULL OR rerun_from_phase IN ("
            "'claim_facts_extraction', 'fact_validation', 'coverage_assessment', "
            "'exclusion_assessment', 'obligation_assessment', 'claim_calculation', "
            "'claim_recommendation')",
            name="ck_assessment_versions_rerun_phase",
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claim_agent_states.claim_id"],
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["assessment_versions.assessment_version_id"],
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.ForeignKeyConstraint(
            ["trigger_reference_id"],
            ["claim_events.event_id"],
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("assessment_version_id"),
        sa.UniqueConstraint(
            "claim_id", "version_number", name="uq_assessment_versions_claim_number"
        ),
    )
    op.create_index(
        "ix_assessment_versions_claim_number",
        "assessment_versions",
        ["claim_id", "version_number"],
    )
    op.create_index(
        "ix_assessment_versions_parent",
        "assessment_versions",
        ["parent_version_id"],
    )
    op.create_index(
        "ix_assessment_versions_trigger_reference",
        "assessment_versions",
        ["trigger_reference_id"],
    )
    op.execute("""
        CREATE FUNCTION reject_assessment_version_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'assessment_versions is append-only';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER trg_assessment_versions_append_only
        BEFORE UPDATE OR DELETE ON assessment_versions
        FOR EACH ROW EXECUTE FUNCTION reject_assessment_version_mutation()
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_assessment_versions_append_only "
        "ON assessment_versions"
    )
    op.execute("DROP FUNCTION IF EXISTS reject_assessment_version_mutation()")
    op.drop_index(
        "ix_assessment_versions_trigger_reference", table_name="assessment_versions"
    )
    op.drop_index("ix_assessment_versions_parent", table_name="assessment_versions")
    op.drop_index("ix_assessment_versions_claim_number", table_name="assessment_versions")
    op.drop_table("assessment_versions")
