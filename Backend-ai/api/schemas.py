"""HTTP request and response schemas for persisted knowledge data."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    knowledge_base_id: uuid.UUID
    name: str
    source_format: str
    status: str
    page_count: int | None
    outline: list[dict[str, Any]]
    document_metadata: dict[str, Any]
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_key: str
    text: str
    chunk_type: str
    chunk_order: int
    token_count: int
    section_path: list[str]
    source_file: str
    source_format: str
    source_block_orders: list[int]
    page_numbers: list[int]
    positions: list[list[float]]
    sheet: str | None
    row_start: int | None
    row_end: int | None
    email_metadata: dict[str, Any]
    context_above: str
    context_below: str


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)
    document_id: uuid.UUID | None = None
    source_format: str | None = Field(default=None, max_length=32)
    page_number: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=500)
    sheet: str | None = Field(default=None, max_length=255)
    retrieval_mode: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"


class RetrievalContextChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    chunk_type: str
    source_file: str
    source_format: str
    page_numbers: list[int]
    section_path: list[str]
    sheet: str | None
    row_start: int | None
    row_end: int | None
    email_metadata: dict[str, Any]
    added_context: bool = True
    context_reasons: list[str]
    anchor_chunk_id: uuid.UUID


class SemanticSearchResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    score: float
    chunk_type: str
    source_file: str
    source_format: str
    page_numbers: list[int]
    section_path: list[str]
    sheet: str | None
    row_start: int | None
    row_end: int | None
    email_metadata: dict[str, Any]
    rank: int
    retrieval_method: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    pre_rerank_rank: int | None = None
    reranker_score: float | None = None
    context_chunks: list[RetrievalContextChunk] = Field(default_factory=list)


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[SemanticSearchResult]


class GroundedAskRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class GroundedCitation(BaseModel):
    evidence_id: str
    chunk_id: uuid.UUID
    source_file: str
    page_numbers: list[int]
    section_path: list[str]
    added_context: bool
    anchor_chunk_id: uuid.UUID | None = None


class GroundedAskResponse(BaseModel):
    query: str
    answer: str
    evidence_ids: list[str]
    insufficient_evidence: bool
    citations: list[GroundedCitation]
