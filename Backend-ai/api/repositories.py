"""Minimal asynchronous CRUD repositories for persistence models."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Text, cast, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.models import (
    AssessmentVersionRecord,
    ClaimAgentStateRecord,
    ClaimEventRecord,
    Chunk,
    Document,
    KnowledgeBase,
)


class AgentStateRevisionConflictError(RuntimeError):
    """The persisted state changed after the caller loaded it."""


class ClaimAgentStateRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, **values: Any) -> ClaimAgentStateRecord:
        record = ClaimAgentStateRecord(**values)
        self.session.add(record)
        await self.session.flush()
        return record

    async def get(self, claim_id: uuid.UUID) -> ClaimAgentStateRecord | None:
        return await self.session.get(ClaimAgentStateRecord, claim_id)

    async def update(
        self,
        *,
        claim_id: uuid.UUID,
        expected_revision: int,
        values: dict[str, Any],
    ) -> ClaimAgentStateRecord:
        statement = (
            update(ClaimAgentStateRecord)
            .where(
                ClaimAgentStateRecord.claim_id == claim_id,
                ClaimAgentStateRecord.revision == expected_revision,
            )
            .values(
                **values,
                revision=ClaimAgentStateRecord.revision + 1,
                updated_at=func.now(),
            )
            .returning(ClaimAgentStateRecord)
        )
        record = (await self.session.execute(statement)).scalar_one_or_none()
        if record is not None:
            return record
        if await self.get(claim_id) is None:
            raise LookupError(f"No persisted agent state for claim {claim_id}")
        raise AgentStateRevisionConflictError(
            f"Agent state for claim {claim_id} no longer has revision {expected_revision}"
        )


class ClaimEventRepository:
    """Append and read claim events; mutation methods intentionally do not exist."""

    _FACT_EVENT_TYPES = (
        "claim_fact_extracted",
        "claim_fact_updated",
        "user_answer_applied",
    )

    def __init__(self, session: AsyncSession):
        self.session = session

    async def append_event(self, **values: Any) -> ClaimEventRecord:
        return (await self.append_events([values]))[0]

    async def append_events(self, values: Sequence[dict[str, Any]]) -> list[ClaimEventRecord]:
        records = [ClaimEventRecord(**item) for item in values]
        self.session.add_all(records)
        await self.session.flush()
        return records

    async def list_events(self, claim_id: uuid.UUID) -> Sequence[ClaimEventRecord]:
        result = await self.session.scalars(
            select(ClaimEventRecord)
            .where(ClaimEventRecord.claim_id == claim_id)
            .order_by(ClaimEventRecord.created_at, ClaimEventRecord.event_id)
        )
        return result.all()

    async def list_fact_history(
        self,
        claim_id: uuid.UUID,
        field_path: str,
    ) -> Sequence[ClaimEventRecord]:
        result = await self.session.scalars(
            select(ClaimEventRecord)
            .where(
                ClaimEventRecord.claim_id == claim_id,
                ClaimEventRecord.field_path == field_path,
                ClaimEventRecord.event_type.in_(self._FACT_EVENT_TYPES),
            )
            .order_by(ClaimEventRecord.created_at, ClaimEventRecord.event_id)
        )
        return result.all()

    async def get_event(
        self, claim_id: uuid.UUID, event_id: uuid.UUID
    ) -> ClaimEventRecord | None:
        return await self.session.scalar(
            select(ClaimEventRecord).where(
                ClaimEventRecord.claim_id == claim_id,
                ClaimEventRecord.event_id == event_id,
            )
        )

    async def list_context_events(
        self,
        claim_id: uuid.UUID,
        *,
        event_types: Sequence[str] = (),
        related_phases: Sequence[str] = (),
        field_path_patterns: Sequence[str] = (),
        missing_information_ids: Sequence[uuid.UUID] = (),
        limit: int,
    ) -> Sequence[ClaimEventRecord]:
        relevance = []
        if related_phases:
            relevance.append(ClaimEventRecord.related_phase.in_(related_phases))
        relevance.extend(
            ClaimEventRecord.field_path.ilike(pattern)
            for pattern in field_path_patterns
        )
        if missing_information_ids:
            relevance.append(
                ClaimEventRecord.missing_information_id.in_(missing_information_ids)
            )
        statement = select(ClaimEventRecord).where(
            ClaimEventRecord.claim_id == claim_id
        )
        if event_types:
            statement = statement.where(ClaimEventRecord.event_type.in_(event_types))
        if relevance:
            statement = statement.where(or_(*relevance))
        result = await self.session.scalars(
            statement.order_by(
                ClaimEventRecord.created_at.desc(),
                ClaimEventRecord.event_id.desc(),
            ).limit(limit)
        )
        return result.all()

    async def list_events_before(
        self,
        claim_id: uuid.UUID,
        *,
        anchor_time,
        anchor_id: uuid.UUID,
        limit: int,
    ) -> Sequence[ClaimEventRecord]:
        result = await self.session.scalars(
            select(ClaimEventRecord)
            .where(
                ClaimEventRecord.claim_id == claim_id,
                or_(
                    ClaimEventRecord.created_at < anchor_time,
                    (
                        (ClaimEventRecord.created_at == anchor_time)
                        & (ClaimEventRecord.event_id < anchor_id)
                    ),
                ),
            )
            .order_by(
                ClaimEventRecord.created_at.desc(),
                ClaimEventRecord.event_id.desc(),
            )
            .limit(limit)
        )
        return result.all()

    async def list_events_after(
        self,
        claim_id: uuid.UUID,
        *,
        anchor_time,
        anchor_id: uuid.UUID,
        limit: int,
    ) -> Sequence[ClaimEventRecord]:
        result = await self.session.scalars(
            select(ClaimEventRecord)
            .where(
                ClaimEventRecord.claim_id == claim_id,
                or_(
                    ClaimEventRecord.created_at > anchor_time,
                    (
                        (ClaimEventRecord.created_at == anchor_time)
                        & (ClaimEventRecord.event_id > anchor_id)
                    ),
                ),
            )
            .order_by(
                ClaimEventRecord.created_at.asc(),
                ClaimEventRecord.event_id.asc(),
            )
            .limit(limit)
        )
        return result.all()

    async def search_text_events(
        self,
        claim_id: uuid.UUID,
        *,
        query: str,
        event_types: Sequence[str],
        limit: int,
    ) -> Sequence[ClaimEventRecord]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        result = await self.session.scalars(
            select(ClaimEventRecord)
            .where(
                ClaimEventRecord.claim_id == claim_id,
                ClaimEventRecord.event_type.in_(event_types),
                ClaimEventRecord.raw_answer.ilike(pattern, escape="\\"),
            )
            .order_by(
                ClaimEventRecord.created_at.desc(),
                ClaimEventRecord.event_id.desc(),
            )
            .limit(limit)
        )
        return result.all()


class AssessmentVersionRepository:
    """Create and read immutable versions; mutation methods intentionally do not exist."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def acquire_claim_lock(self, claim_id: uuid.UUID) -> None:
        # A transaction-scoped PostgreSQL advisory lock also works before the
        # deferred claim-state FK has been inserted in the same transaction.
        await self.session.execute(
            select(func.pg_advisory_xact_lock(func.hashtextextended(str(claim_id), 0)))
        )

    async def create_version(self, **values: Any) -> AssessmentVersionRecord:
        record = AssessmentVersionRecord(**values)
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_latest_version(
        self, claim_id: uuid.UUID
    ) -> AssessmentVersionRecord | None:
        return await self.session.scalar(
            select(AssessmentVersionRecord)
            .where(AssessmentVersionRecord.claim_id == claim_id)
            .order_by(AssessmentVersionRecord.version_number.desc())
            .limit(1)
        )

    async def list_versions(
        self, claim_id: uuid.UUID
    ) -> Sequence[AssessmentVersionRecord]:
        result = await self.session.scalars(
            select(AssessmentVersionRecord)
            .where(AssessmentVersionRecord.claim_id == claim_id)
            .order_by(AssessmentVersionRecord.version_number)
        )
        return result.all()

    async def get_version(
        self, version_id: uuid.UUID
    ) -> AssessmentVersionRecord | None:
        return await self.session.get(AssessmentVersionRecord, version_id)

    async def list_context_versions(
        self,
        claim_id: uuid.UUID,
        *,
        related_phases: Sequence[str],
        limit: int,
    ) -> Sequence[AssessmentVersionRecord]:
        relevance = [
            AssessmentVersionRecord.rerun_from_phase.in_(related_phases)
        ] if related_phases else []
        relevance.extend(
            AssessmentVersionRecord.completed_phases.contains([phase])
            for phase in related_phases
        )
        statement = select(AssessmentVersionRecord).where(
            AssessmentVersionRecord.claim_id == claim_id
        )
        if relevance:
            statement = statement.where(or_(*relevance))
        result = await self.session.scalars(
            statement.order_by(
                AssessmentVersionRecord.version_number.desc(),
                AssessmentVersionRecord.assessment_version_id.desc(),
            ).limit(limit)
        )
        return result.all()

    async def search_text_versions(
        self,
        claim_id: uuid.UUID,
        *,
        query: str,
        limit: int,
    ) -> Sequence[AssessmentVersionRecord]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        result = await self.session.scalars(
            select(AssessmentVersionRecord)
            .where(
                AssessmentVersionRecord.claim_id == claim_id,
                or_(
                    cast(AssessmentVersionRecord.phase_outputs, Text).ilike(
                        pattern, escape="\\"
                    ),
                    cast(AssessmentVersionRecord.recommendation, Text).ilike(
                        pattern, escape="\\"
                    ),
                ),
            )
            .order_by(
                AssessmentVersionRecord.version_number.desc(),
                AssessmentVersionRecord.assessment_version_id.desc(),
            )
            .limit(limit)
        )
        return result.all()

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
