"""Persist parsed documents, structured chunks, and embeddings."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from api.document_chunker import Chunk as StructuredChunk
from api.document_chunker import chunk_document
from api.document_parser import ParsedDocument, parse_document
from api.models import Document, EMBEDDING_DIMENSION
from api.repositories import ChunkRepository, DocumentRepository, KnowledgeBaseRepository


class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str]): ...


def _embedding_rows(values) -> list[list[float]]:
    rows = values.tolist() if hasattr(values, "tolist") else values
    return [[float(value) for value in row] for row in rows]


def chunk_record(chunk: StructuredChunk, document_id: uuid.UUID, embedding: list[float]) -> dict:
    """Map the existing in-memory Chunk without changing chunking behavior."""
    return {
        "document_id": document_id,
        "chunk_key": chunk.chunk_id,
        "text": chunk.text,
        "chunk_type": chunk.type,
        "chunk_order": chunk.order,
        "token_count": chunk.token_count,
        "section_path": chunk.section_path,
        "source_file": chunk.source_file,
        "source_format": chunk.source_format,
        "source_block_orders": chunk.source_block_orders,
        "page_numbers": chunk.page_numbers,
        "positions": chunk.positions,
        "sheet": chunk.sheet,
        "row_start": chunk.row_start,
        "row_end": chunk.row_end,
        "email_metadata": chunk.email_metadata,
        "context_above": chunk.context_above,
        "context_below": chunk.context_below,
        "embedding": embedding,
    }


class DocumentIngestionService:
    def __init__(self, session: AsyncSession, embedding_model: EmbeddingModel):
        self.session = session
        self.embedding_model = embedding_model
        self.knowledge_bases = KnowledgeBaseRepository(session)
        self.documents = DocumentRepository(session)
        self.chunks = ChunkRepository(session)

    async def ingest(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        file_path: str | Path,
        original_filename: str,
    ) -> Document:
        knowledge_base = await self.knowledge_bases.get(knowledge_base_id)
        if knowledge_base is None:
            raise LookupError("Knowledge base not found")

        source_format = Path(original_filename).suffix.lower().lstrip(".")
        document = await self.documents.create(
            knowledge_base_id=knowledge_base_id,
            name=original_filename,
            source_format=source_format,
            status="pending",
        )
        await self.session.commit()

        await self.documents.update(document, status="processing", error_message=None)
        await self.session.commit()

        try:
            parsed = await asyncio.to_thread(parse_document, file_path)
            parsed.name = original_filename
            chunks = await asyncio.to_thread(chunk_document, parsed)
            if not chunks:
                raise ValueError("Document produced no chunks")

            raw_embeddings = await asyncio.to_thread(
                self.embedding_model.encode, [chunk.text for chunk in chunks]
            )
            embeddings = _embedding_rows(raw_embeddings)
            self._validate_embeddings(chunks, embeddings)

            await self.documents.update(
                document,
                page_count=parsed.page_count,
                outline=parsed.outline,
                document_metadata=parsed.metadata,
            )
            await self.chunks.create_many(
                [
                    chunk_record(chunk, document.id, embedding)
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ]
            )
            await self.documents.update(document, status="completed", error_message=None)
            await self.session.commit()
            return document
        except Exception as error:
            await self.session.rollback()
            failed_document = await self.documents.get(document.id)
            if failed_document is not None:
                await self.documents.update(
                    failed_document,
                    status="failed",
                    error_message=str(error)[:2000],
                )
                await self.session.commit()
            raise

    @staticmethod
    def _validate_embeddings(chunks: list[StructuredChunk], embeddings: list[list[float]]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")
        if any(len(embedding) != EMBEDDING_DIMENSION for embedding in embeddings):
            raise ValueError(f"Every embedding must contain {EMBEDDING_DIMENSION} values")
