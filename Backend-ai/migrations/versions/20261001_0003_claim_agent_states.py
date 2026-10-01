"""Create current claim agent-state persistence.

Revision ID: 20261001_0003
Revises: 20260923_0002
Create Date: 2026-10-01
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "20261001_0003"
down_revision: str | None = "20260923_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    empty_list = sa.text("'[]'::jsonb")
    op.create_table(
        "claim_agent_states",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("knowledge_base_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("claim", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("facts", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("missing_information", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("applicable_policy", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("coverage_assessments", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("exclusion_assessments", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("obligation_assessments", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("calculation_results", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("recommendation", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("current_phase", sa.String(length=64), nullable=True),
        sa.Column("next_action", sa.String(length=32), nullable=False),
        sa.Column("completed_phases", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("provider_errors", postgresql.JSONB(astext_type=sa.Text()), server_default=empty_list, nullable=False),
        sa.Column("rerun_failure_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("retrieval_limit", sa.Integer(), server_default="8", nullable=False),
        sa.Column("claimant_reference_required", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("policy_reference_required", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("state_schema_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_claim_agent_states_revision_positive"),
        sa.CheckConstraint("state_schema_version >= 1", name="ck_claim_agent_states_schema_version_positive"),
        sa.CheckConstraint("rerun_failure_count >= 0", name="ck_claim_agent_states_rerun_failure_count_nonnegative"),
        sa.CheckConstraint("retrieval_limit BETWEEN 1 AND 20", name="ck_claim_agent_states_retrieval_limit"),
        sa.CheckConstraint(
            "next_action IN ('continue_assessment', 'ask_for_information', 'rerun_phase', 'human_review', 'completed')",
            name="ck_claim_agent_states_next_action",
        ),
        sa.CheckConstraint(
            "current_phase IS NULL OR current_phase IN ('claim_facts_extraction', 'fact_validation', 'coverage_assessment', 'exclusion_assessment', 'obligation_assessment', 'claim_calculation', 'claim_recommendation')",
            name="ck_claim_agent_states_current_phase",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("claim_id"),
    )
    op.create_index(
        op.f("ix_claim_agent_states_knowledge_base_id"),
        "claim_agent_states",
        ["knowledge_base_id"],
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_claim_agent_states_knowledge_base_id"),
        table_name="claim_agent_states",
    )
    op.drop_table("claim_agent_states")
