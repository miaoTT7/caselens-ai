import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.main import ask_question
from api.schemas import GroundedAskRequest, GroundedCitation


class AskEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_ask_uses_hybrid_rerank_and_returns_mapped_citations(self):
        chunk_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        result = SimpleNamespace(chunk_id=chunk_id)
        citation = GroundedCitation(
            evidence_id="[E1]",
            chunk_id=chunk_id,
            source_file="policy.pdf",
            page_numbers=[22],
            section_path=["F2 Bicycles"],
            added_context=False,
        )
        retrieval = AsyncMock(return_value=[result])
        llm_answer = SimpleNamespace(
            answer="Bicycle theft is covered subject to a lock condition.",
            evidence_ids=["[E1]"],
            insufficient_evidence=False,
        )
        generation = AsyncMock(return_value=(llm_answer, [citation]))

        with (
            patch("api.main.SemanticRetrievalService.search", retrieval),
            patch("api.main.GroundedAnswerService.answer", generation),
            patch("api.main.GroundedAnswerService.__init__", return_value=None),
        ):
            response = await ask_question(
                GroundedAskRequest(knowledge_base_id=kb_id, query=" bicycle theft ", limit=5),
                AsyncMock(),
            )

        request = retrieval.await_args.args[1]
        self.assertEqual(request.retrieval_mode, "hybrid_rerank")
        self.assertEqual(request.limit, 5)
        self.assertEqual(response.evidence_ids, ["[E1]"])
        self.assertEqual(response.citations[0].chunk_id, chunk_id)


if __name__ == "__main__":
    unittest.main()
