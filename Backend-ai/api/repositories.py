"""Minimal asynchronous CRUD repositories for persistence models."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import Chunk, Document, KnowledgeBase


class KnowledgeBaseRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, *, name: str, description: str | None = None) -> KnowledgeBase:
        knowledge_base = KnowledgeBase(name=name, description=description)
        self.session.add(knowledge_base)
        await self.session.flush()
        return knowledge_base

    async def get(self, knowledge_base_id: uuid.UUID) -> KnowledgeBase | None:
        return await self.session.get(KnowledgeBase, knowledge_base_id)

    async def get_by_name(self, name: str) -> KnowledgeBase | None:
        return await self.session.scalar(select(KnowledgeBase).where(KnowledgeBase.name == name))

    async def list(self) -> Sequence[KnowledgeBase]:
        result = await self.session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.created_at))
        return result.all()

    async def update(self, knowledge_base: KnowledgeBase, **values: Any) -> KnowledgeBase:
        for key, value in values.items():
            setattr(knowledge_base, key, value)
        await self.session.flush()
        return knowledge_base

    async def delete(self, knowledge_base: KnowledgeBase) -> None:
        await self.session.delete(knowledge_base)
        await self.session.flush()


class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **values: Any) -> Document:
        document = Document(**values)
        self.session.add(document)
        await self.session.flush()
        return document

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list_for_knowledge_base(self, knowledge_base_id: uuid.UUID) -> Sequence[Document]:
        result = await self.session.scalars(
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at)
        )
        return result.all()

    async def update(self, document: Document, **values: Any) -> Document:
        for key, value in values.items():
            setattr(document, key, value)
        await self.session.flush()
        return document

    async def delete(self, document: Document) -> None:
        await self.session.delete(document)
        await self.session.flush()


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **values: Any) -> Chunk:
        chunk = Chunk(**values)
        self.session.add(chunk)
        await self.session.flush()
        return chunk

    async def create_many(self, values: Sequence[dict[str, Any]]) -> list[Chunk]:
        chunks = [Chunk(**item) for item in values]
        self.session.add_all(chunks)
        await self.session.flush()
        return chunks

    async def get(self, chunk_id: uuid.UUID) -> Chunk | None:
        return await self.session.get(Chunk, chunk_id)

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[Chunk]:
        result = await self.session.scalars(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_order)
        )
        return result.all()

    async def list_searchable(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        document_id: uuid.UUID | None = None,
        source_format: str | None = None,
        page_number: int | None = None,
        section: str | None = None,
        sheet: str | None = None,
    ) -> Sequence[Chunk]:
        statement = (
            select(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.status == "completed",
            )
            .order_by(Document.created_at, Chunk.chunk_order)
        )
        if document_id is not None:
            statement = statement.where(Chunk.document_id == document_id)
        if source_format is not None:
            statement = statement.where(Document.source_format == source_format)
        if page_number is not None:
            statement = statement.where(Chunk.page_numbers.contains([page_number]))
        if section is not None:
            statement = statement.where(Chunk.section_path.contains([section]))
        if sheet is not None:
            statement = statement.where(Chunk.sheet == sheet)
        return (await self.session.scalars(statement)).all()

    async def search_by_vector(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query_embedding: list[float],
        limit: int,
        document_id: uuid.UUID | None = None,
        source_format: str | None = None,
        page_number: int | None = None,
        section: str | None = None,
        sheet: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return nearest chunks and their cosine distance within one knowledge base."""
        distance = Chunk.embedding.cosine_distance(query_embedding)
        statement = (
            select(Chunk, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.status == "completed",
                Chunk.embedding.is_not(None),
            )
        )
        if document_id is not None:
            statement = statement.where(Chunk.document_id == document_id)
        if source_format is not None:
            statement = statement.where(Document.source_format == source_format)
        if page_number is not None:
            statement = statement.where(Chunk.page_numbers.contains([page_number]))
        if section is not None:
            statement = statement.where(Chunk.section_path.contains([section]))
        if sheet is not None:
            statement = statement.where(Chunk.sheet == sheet)

        rows = await self.session.execute(statement.order_by(distance).limit(limit))
        return [(chunk, float(value)) for chunk, value in rows.all()]

    async def update(self, chunk: Chunk, **values: Any) -> Chunk:
        for key, value in values.items():
            setattr(chunk, key, value)
        await self.session.flush()
        return chunk

    async def delete(self, chunk: Chunk) -> None:
        await self.session.delete(chunk)
        await self.session.flush()
