"""Persistence models for CaseLens knowledge bases and parsed documents."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


EMBEDDING_DIMENSION = 384


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KnowledgeBase(TimestampMixin, Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan", passive_deletes=True
    )


class Document(TimestampMixin, Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_format: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", server_default="pending")
    page_count: Mapped[int | None] = mapped_column(Integer)
    outline: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    document_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(TimestampMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_key", name="uq_chunks_document_chunk_key"),
        UniqueConstraint("document_id", "chunk_order", name="uq_chunks_document_order"),
        Index("ix_chunks_document_id", "document_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_key: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_order: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    source_format: Mapped[str] = mapped_column(String(32), nullable=False)
    source_block_orders: Mapped[list[int]] = mapped_column(JSONB, nullable=False, default=list)
    page_numbers: Mapped[list[int]] = mapped_column(JSONB, nullable=False, default=list)
    positions: Mapped[list[list[float]]] = mapped_column(JSONB, nullable=False, default=list)
    sheet: Mapped[str | None] = mapped_column(String(255))
    row_start: Mapped[int | None] = mapped_column(Integer)
    row_end: Mapped[int | None] = mapped_column(Integer)
    email_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    context_above: Mapped[str] = mapped_column(Text, nullable=False, default="")
    context_below: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list[float] | None] = mapped_column(VECTOR(EMBEDDING_DIMENSION))

    document: Mapped[Document] = relationship(back_populates="chunks")


class ClaimAgentStateRecord(TimestampMixin, Base):
    """The single current, resumable agent-state snapshot for one claim."""

    __tablename__ = "claim_agent_states"
    __table_args__ = (
        CheckConstraint("revision >= 1", name="ck_claim_agent_states_revision_positive"),
        CheckConstraint(
            "state_schema_version >= 1",
            name="ck_claim_agent_states_schema_version_positive",
        ),
        CheckConstraint(
            "rerun_failure_count >= 0",
            name="ck_claim_agent_states_rerun_failure_count_nonnegative",
        ),
        CheckConstraint(
            "retrieval_limit BETWEEN 1 AND 20",
            name="ck_claim_agent_states_retrieval_limit",
        ),
        CheckConstraint(
            "next_action IN ('continue_assessment', 'ask_for_information', "
            "'rerun_phase', 'human_review', 'completed')",
            name="ck_claim_agent_states_next_action",
        ),
        CheckConstraint(
            "current_phase IS NULL OR current_phase IN ("
            "'claim_facts_extraction', 'fact_validation', 'coverage_assessment', "
            "'exclusion_assessment', 'obligation_assessment', 'claim_calculation', "
            "'claim_recommendation')",
            name="ck_claim_agent_states_current_phase",
        ),
    )

    claim_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True
    )
    claim: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    facts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    missing_information: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    applicable_policy: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    coverage_assessments: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    exclusion_assessments: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    obligation_assessments: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    calculation_results: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    recommendation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    current_phase: Mapped[str | None] = mapped_column(String(64))
    next_action: Mapped[str] = mapped_column(String(32), nullable=False)
    completed_phases: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    provider_errors: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    rerun_failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieval_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    claimant_reference_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    policy_reference_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    state_schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class ClaimEventRecord(Base):
    """An immutable audit event describing an observed claim-state transition."""

    __tablename__ = "claim_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('fnol_submitted', 'claim_fact_extracted', "
            "'claim_fact_updated', 'missing_information_created', "
            "'user_answer_received', 'user_answer_applied', 'provider_error', "
            "'phase_completed', 'selective_rerun_started', "
            "'selective_rerun_completed', 'human_handoff_created')",
            name="ck_claim_events_event_type",
        ),
        CheckConstraint(
            "actor_type IN ('user', 'agent', 'system', 'provider', 'human_reviewer')",
            name="ck_claim_events_actor_type",
        ),
        CheckConstraint(
            "related_phase IS NULL OR related_phase IN ("
            "'claim_facts_extraction', 'fact_validation', 'coverage_assessment', "
            "'exclusion_assessment', 'obligation_assessment', 'claim_calculation', "
            "'claim_recommendation')",
            name="ck_claim_events_related_phase",
        ),
        Index("ix_claim_events_claim_created", "claim_id", "created_at", "event_id"),
        Index(
            "ix_claim_events_claim_field_created",
            "claim_id",
            "field_path",
            "created_at",
        ),
    )

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "claim_agent_states.claim_id",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=False,
    )
    assessment_session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    field_path: Mapped[str | None] = mapped_column(Text)
    old_value: Mapped[Any | None] = mapped_column(JSONB)
    new_value: Mapped[Any | None] = mapped_column(JSONB)
    raw_answer: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[Any | None] = mapped_column(JSONB)
    missing_information_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    evidence_refs: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    related_phase: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AssessmentVersionRecord(Base):
    """An immutable structured snapshot of one stable claim assessment."""

    __tablename__ = "assessment_versions"
    __table_args__ = (
        UniqueConstraint(
            "claim_id", "version_number", name="uq_assessment_versions_claim_number"
        ),
        CheckConstraint(
            "version_number >= 1", name="ck_assessment_versions_number_positive"
        ),
        CheckConstraint(
            "trigger_type IN ('initial_assessment', 'user_follow_up', "
            "'selective_rerun', 'provider_retry', 'manual_review')",
            name="ck_assessment_versions_trigger_type",
        ),
        CheckConstraint(
            "status IN ('incomplete', 'completed', 'needs_information', "
            "'needs_human_review', 'failed')",
            name="ck_assessment_versions_status",
        ),
        CheckConstraint(
            "rerun_from_phase IS NULL OR rerun_from_phase IN ("
            "'claim_facts_extraction', 'fact_validation', 'coverage_assessment', "
            "'exclusion_assessment', 'obligation_assessment', 'claim_calculation', "
            "'claim_recommendation')",
            name="ck_assessment_versions_rerun_phase",
        ),
        Index(
            "ix_assessment_versions_claim_number",
            "claim_id",
            "version_number",
        ),
        Index("ix_assessment_versions_parent", "parent_version_id"),
        Index("ix_assessment_versions_trigger_reference", "trigger_reference_id"),
    )

    assessment_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey(
            "claim_agent_states.claim_id",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=False,
    )
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "assessment_versions.assessment_version_id",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        )
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_reference_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey(
            "claim_events.event_id",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        )
    )
    rerun_from_phase: Mapped[str | None] = mapped_column(String(64))
    completed_phases: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    phase_outputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    recommendation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
