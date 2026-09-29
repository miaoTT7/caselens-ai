import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from api.document_chunker import Chunk
from api.document_parser import ParsedDocument
from api.ingestion import DocumentIngestionService, chunk_record
from api.models import EMBEDDING_DIMENSION


class IngestionMappingTests(unittest.TestCase):
    def test_chunk_record_preserves_structured_fields(self):
        document_id = uuid.uuid4()
        chunk = Chunk(
            chunk_id="stable-key",
            text="Coverage applies.",
            type="text",
            order=2,
            token_count=3,
            section_path=["Coverage"],
            source_file="policy.pdf",
            source_format="pdf",
            source_block_orders=[4],
            page_numbers=[2],
            positions=[[1.0, 2.0, 3.0, 4.0]],
        )
        embedding = [0.0] * EMBEDDING_DIMENSION

        values = chunk_record(chunk, document_id, embedding)

        self.assertEqual(values["chunk_key"], "stable-key")
        self.assertEqual(values["chunk_order"], 2)
        self.assertEqual(values["section_path"], ["Coverage"])
        self.assertEqual(values["page_numbers"], [2])
        self.assertEqual(values["embedding"], embedding)


class IngestionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingestion_persists_chunks_and_completes_document(self):
        session = AsyncMock()
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.1] * EMBEDDING_DIMENSION]
        service = DocumentIngestionService(session, embedding_model)
        document = SimpleNamespace(id=uuid.uuid4(), status="pending")
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        service.documents.create = AsyncMock(return_value=document)
        service.documents.update = AsyncMock(return_value=document)
        service.chunks.create_many = AsyncMock(return_value=[])
        parsed = ParsedDocument(name="temporary.pdf", format="pdf", blocks=[], page_count=1)
        chunk = Chunk(
            chunk_id="chunk-1",
            text="Policy coverage",
            type="text",
            order=0,
            token_count=2,
            section_path=["Coverage"],
            source_file="policy.pdf",
            source_format="pdf",
            source_block_orders=[0],
        )

        with (
            patch("api.ingestion.parse_document", return_value=parsed),
            patch("api.ingestion.chunk_document", return_value=[chunk]),
        ):
            result = await service.ingest(
                knowledge_base_id=uuid.uuid4(),
                file_path="temporary.pdf",
                original_filename="policy.pdf",
            )

        self.assertIs(result, document)
        self.assertEqual(session.commit.await_count, 3)
        saved = service.chunks.create_many.await_args.args[0]
        self.assertEqual(saved[0]["chunk_key"], "chunk-1")
        self.assertEqual(len(saved[0]["embedding"]), EMBEDDING_DIMENSION)
        self.assertEqual(service.documents.update.await_args.kwargs["status"], "completed")

    async def test_failure_rolls_back_chunks_and_marks_document_failed(self):
        session = AsyncMock()
        service = DocumentIngestionService(session, Mock())
        document = SimpleNamespace(id=uuid.uuid4(), status="pending")
        failed_document = SimpleNamespace(id=document.id, status="processing")
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        service.documents.create = AsyncMock(return_value=document)
        service.documents.update = AsyncMock(return_value=document)
        service.documents.get = AsyncMock(return_value=failed_document)

        with patch("api.ingestion.parse_document", side_effect=ValueError("invalid document")):
            with self.assertRaisesRegex(ValueError, "invalid document"):
                await service.ingest(
                    knowledge_base_id=uuid.uuid4(),
                    file_path="broken.pdf",
                    original_filename="broken.pdf",
                )

        session.rollback.assert_awaited_once()
        self.assertEqual(session.commit.await_count, 3)
        self.assertEqual(service.documents.update.await_args.kwargs["status"], "failed")
        self.assertEqual(service.documents.update.await_args.kwargs["error_message"], "invalid document")

    async def test_missing_knowledge_base_stops_before_document_creation(self):
        service = DocumentIngestionService(AsyncMock(), Mock())
        service.knowledge_bases.get = AsyncMock(return_value=None)
        service.documents.create = AsyncMock()

        with self.assertRaisesRegex(LookupError, "Knowledge base not found"):
            await service.ingest(
                knowledge_base_id=uuid.uuid4(),
                file_path="policy.pdf",
                original_filename="policy.pdf",
            )

        service.documents.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
