import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from api.models import EMBEDDING_DIMENSION
from api.retrieval import SemanticRetrievalService
from api.schemas import SemanticSearchRequest


class SemanticRetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_embeds_query_and_returns_source_metadata(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * EMBEDDING_DIMENSION]
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        chunk = SimpleNamespace(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            text="Water damage is covered.",
            chunk_type="text",
            source_file="policy.pdf",
            source_format="pdf",
            page_numbers=[8],
            section_path=["Property Coverage"],
            sheet=None,
            row_start=None,
            row_end=None,
            email_metadata={},
            chunk_order=4,
        )
        service.chunks.search_by_vector = AsyncMock(return_value=[(chunk, 0.13)])
        service.chunks.list_for_document = AsyncMock(return_value=[chunk])
        request = SemanticSearchRequest(query=" water damage ", limit=3, page_number=8)

        results = await service.search(uuid.uuid4(), request)

        embedding_model.encode.assert_called_once_with(["water damage"])
        self.assertAlmostEqual(results[0].score, 0.87)
        self.assertEqual(results[0].page_numbers, [8])
        self.assertEqual(results[0].section_path, ["Property Coverage"])
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].context_chunks, [])
        self.assertEqual(service.chunks.search_by_vector.await_args.kwargs["limit"], 3)
        self.assertEqual(service.chunks.search_by_vector.await_args.kwargs["page_number"], 8)

    async def test_missing_knowledge_base_does_not_embed_query(self):
        embedding_model = Mock()
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=None)

        with self.assertRaisesRegex(LookupError, "Knowledge base not found"):
            await service.search(uuid.uuid4(), SemanticSearchRequest(query="coverage"))

        embedding_model.encode.assert_not_called()

    async def test_invalid_embedding_dimension_is_rejected(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * 10]
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))

        with self.assertRaisesRegex(ValueError, "384"):
            await service.search(uuid.uuid4(), SemanticSearchRequest(query="coverage"))

    async def test_context_expansion_keeps_dense_rank_and_marks_related_chunks(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * EMBEDDING_DIMENSION]
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        document_id = uuid.uuid4()

        def chunk(order, section):
            return SimpleNamespace(
                id=uuid.uuid4(),
                document_id=document_id,
                text=f"Chunk {order}",
                chunk_type="text",
                chunk_order=order,
                source_file="policy.pdf",
                source_format="pdf",
                page_numbers=[1],
                section_path=section,
                sheet=None,
                row_start=None,
                row_end=None,
                email_metadata={},
            )

        previous = chunk(1, ["F2 Bicycles", "F2.1 Insured property"])
        anchor = chunk(2, ["F2 Bicycles", "F2.3 Insured risks"])
        next_chunk = chunk(3, ["F2 Bicycles", "F2.4 Exclusions"])
        unrelated = chunk(4, ["F3 Luggage", "F3.1 Insured property"])
        service.chunks.search_by_vector = AsyncMock(return_value=[(anchor, 0.2)])
        service.chunks.list_for_document = AsyncMock(
            return_value=[previous, anchor, next_chunk, unrelated]
        )

        results = await service.search(uuid.uuid4(), SemanticSearchRequest(query="bicycle theft"))

        self.assertEqual(results[0].rank, 1)
        self.assertAlmostEqual(results[0].score, 0.8)
        self.assertEqual(
            [context.chunk_id for context in results[0].context_chunks],
            [previous.id, next_chunk.id],
        )
        self.assertTrue(all(context.added_context for context in results[0].context_chunks))
        self.assertIn("same_parent_section", results[0].context_chunks[0].context_reasons)
        self.assertNotIn(unrelated.id, [context.chunk_id for context in results[0].context_chunks])

    async def test_context_expansion_deduplicates_original_dense_results(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * EMBEDDING_DIMENSION]
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        document_id = uuid.uuid4()

        def chunk(order):
            return SimpleNamespace(
                id=uuid.uuid4(), document_id=document_id, text=f"Chunk {order}",
                chunk_type="text", chunk_order=order, source_file="policy.pdf",
                source_format="pdf", page_numbers=[1], section_path=["Coverage", f"S{order}"],
                sheet=None, row_start=None, row_end=None, email_metadata={},
            )

        first, second, third = chunk(1), chunk(2), chunk(3)
        service.chunks.search_by_vector = AsyncMock(return_value=[(first, 0.1), (second, 0.2)])
        service.chunks.list_for_document = AsyncMock(return_value=[first, second, third])

        results = await service.search(uuid.uuid4(), SemanticSearchRequest(query="coverage"))

        context_ids = [context.chunk_id for result in results for context in result.context_chunks]
        self.assertNotIn(first.id, context_ids)
        self.assertNotIn(second.id, context_ids)
        self.assertEqual(context_ids.count(third.id), 1)

    async def test_broad_parent_only_expands_adjacent_chunks(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * EMBEDDING_DIMENSION]
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        document_id = uuid.uuid4()

        def chunk(order):
            return SimpleNamespace(
                id=uuid.uuid4(), document_id=document_id, text=f"Chunk {order}",
                chunk_type="text", chunk_order=order, source_file="policy.pdf",
                source_format="pdf", page_numbers=[1], section_path=["Contents", f"S{order}"],
                sheet=None, row_start=None, row_end=None, email_metadata={},
            )

        chunks = [chunk(order) for order in range(9)]
        anchor = chunks[4]
        service.chunks.search_by_vector = AsyncMock(return_value=[(anchor, 0.1)])
        service.chunks.list_for_document = AsyncMock(return_value=chunks)

        results = await service.search(uuid.uuid4(), SemanticSearchRequest(query="coverage"))

        self.assertEqual(
            [context.chunk_id for context in results[0].context_chunks],
            [chunks[3].id, chunks[5].id],
        )
        self.assertTrue(
            all("same_parent_section" not in context.context_reasons for context in results[0].context_chunks)
        )

    async def test_hybrid_search_fuses_vector_and_bm25_without_changing_component_ranks(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * EMBEDDING_DIMENSION]
        service = SemanticRetrievalService(AsyncMock(), embedding_model)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        document_id = uuid.uuid4()

        def chunk(order, text):
            return SimpleNamespace(
                id=uuid.uuid4(), document_id=document_id, text=text,
                chunk_type="text", chunk_order=order, source_file="policy.pdf",
                source_format="pdf", page_numbers=[1], section_path=[f"S{order}"],
                sheet=None, row_start=None, row_end=None, email_metadata={},
            )

        vector_first = chunk(1, "General household insurance coverage.")
        shared = chunk(2, "Bicycle loss is insured.")
        keyword_first = chunk(3, "Bicycle theft special conditions bicycle theft.")
        service.chunks.search_by_vector = AsyncMock(
            return_value=[(vector_first, 0.1), (shared, 0.2)]
        )
        service.chunks.list_searchable = AsyncMock(
            return_value=[vector_first, shared, keyword_first]
        )
        service.chunks.list_for_document = AsyncMock(
            return_value=[vector_first, shared, keyword_first]
        )

        results = await service.search(
            uuid.uuid4(),
            SemanticSearchRequest(query="bicycle theft", limit=3, retrieval_mode="hybrid"),
        )

        self.assertEqual(results[0].chunk_id, shared.id)
        self.assertEqual(results[0].retrieval_method, "hybrid")
        self.assertEqual(results[0].vector_rank, 2)
        self.assertIsNotNone(results[0].bm25_rank)
        self.assertEqual(results[0].score, results[0].rrf_score)
        keyword_result = next(result for result in results if result.chunk_id == keyword_first.id)
        self.assertIsNone(keyword_result.vector_rank)
        self.assertEqual(keyword_result.bm25_rank, 1)
        self.assertEqual(
            service.chunks.search_by_vector.await_args.kwargs["limit"],
            20,
        )

    def test_bm25_prefers_exact_keyword_match(self):
        first = SimpleNamespace(id=uuid.uuid4(), text="General insurance coverage", chunk_order=1)
        second = SimpleNamespace(id=uuid.uuid4(), text="Bicycle theft special conditions", chunk_order=2)

        results = SemanticRetrievalService._bm25_search(
            "bicycle theft", [first, second], limit=2
        )

        self.assertEqual(results[0][0].id, second.id)
        self.assertGreater(results[0][1], 0)

    def test_bm25_normalizes_common_english_plurals(self):
        chunk = SimpleNamespace(
            id=uuid.uuid4(),
            text="Bicycles are insured against covered losses.",
            chunk_order=1,
        )

        results = SemanticRetrievalService._bm25_search("bicycle loss", [chunk], limit=1)

        self.assertEqual(results[0][0].id, chunk.id)

    async def test_hybrid_reranker_scores_anchor_chunk_only(self):
        embedding_model = Mock()
        embedding_model.encode.return_value = [[0.2] * EMBEDDING_DIMENSION]
        reranker = Mock()
        service = SemanticRetrievalService(AsyncMock(), embedding_model, reranker)
        service.knowledge_bases.get = AsyncMock(return_value=SimpleNamespace(id=uuid.uuid4()))
        first_document = uuid.uuid4()
        second_document = uuid.uuid4()

        def chunk(identifier, document_id, order, text, section):
            return SimpleNamespace(
                id=identifier, document_id=document_id, text=text,
                chunk_type="text", chunk_order=order, source_file="policy.pdf",
                source_format="pdf", page_numbers=[1], section_path=section,
                sheet=None, row_start=None, row_end=None, email_metadata={},
            )

        first = chunk(uuid.uuid4(), first_document, 1, "Bicycle insurance.", ["F2", "F2.1"])
        context = chunk(uuid.uuid4(), first_document, 2, "Critical policy conditions.", ["F2", "F2.3"])
        second = chunk(uuid.uuid4(), second_document, 1, "General theft insurance.", ["Theft"])
        service.chunks.search_by_vector = AsyncMock(return_value=[(second, 0.1), (first, 0.2)])
        service.chunks.list_searchable = AsyncMock(return_value=[first, context, second])

        async def list_document(document_id):
            return [first, context] if document_id == first_document else [second]

        service.chunks.list_for_document = AsyncMock(side_effect=list_document)

        def predict(pairs, **kwargs):
            self.assertEqual(kwargs["batch_size"], 8)
            self.assertTrue(all("Critical policy conditions" not in document for _, document in pairs))
            return [0.1 if "General theft" in document else 0.9 for _, document in pairs]

        reranker.predict.side_effect = predict

        results = await service.search(
            uuid.uuid4(),
            SemanticSearchRequest(query="bicycle theft", limit=2, retrieval_mode="hybrid_rerank"),
        )

        self.assertEqual(results[0].chunk_id, first.id)
        self.assertEqual(results[0].retrieval_method, "hybrid_rerank")
        self.assertIsNotNone(results[0].pre_rerank_rank)
        self.assertEqual(results[0].reranker_score, 0.9)
        self.assertEqual(results[0].context_chunks[0].chunk_id, context.id)

    def test_reranker_normalizes_out_of_range_logits(self):
        self.assertEqual(
            SemanticRetrievalService._normalize_scores([-2.0, 0.0, 2.0]),
            [0.0, 0.5, 1.0],
        )

    def test_rrf_scores_are_min_max_normalized_for_weighted_fusion(self):
        scores = SemanticRetrievalService._min_max_scores([0.01, 0.02, 0.03])
        for actual, expected in zip(scores, [0.0, 0.5, 1.0], strict=True):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
