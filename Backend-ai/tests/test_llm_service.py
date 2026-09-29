import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.llm_service import GroundedAnswerService
from api.schemas import RetrievalContextChunk, SemanticSearchResult


def search_result():
    anchor_id = uuid.uuid4()
    document_id = uuid.uuid4()
    return SemanticSearchResult(
        chunk_id=anchor_id,
        document_id=document_id,
        text="Bicycle theft is covered when the bicycle is locked.",
        score=0.9,
        chunk_type="text",
        source_file="policy.pdf",
        source_format="pdf",
        page_numbers=[22],
        section_path=["F2 Bicycles"],
        sheet=None,
        row_start=None,
        row_end=None,
        email_metadata={},
        rank=1,
        context_chunks=[
            RetrievalContextChunk(
                chunk_id=uuid.uuid4(),
                document_id=document_id,
                text="The lock must meet the policy security standard.",
                chunk_type="text",
                source_file="policy.pdf",
                source_format="pdf",
                page_numbers=[22],
                section_path=["F2 Bicycles", "Security conditions"],
                sheet=None,
                row_start=None,
                row_end=None,
                email_metadata={},
                context_reasons=["same_parent_section"],
                anchor_chunk_id=anchor_id,
            )
        ],
    )


class GroundedAnswerServiceTests(unittest.IsolatedAsyncioTestCase):
    def client(self, payload):
        client = SimpleNamespace()
        client.chat = SimpleNamespace()
        client.chat.completions = SimpleNamespace()
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )
        )
        return client

    async def test_maps_model_evidence_ids_to_real_context_metadata(self):
        client = self.client(
            {
                "answer": "The bicycle must use a qualifying lock.",
                "evidence_ids": ["[E2]"],
                "insufficient_evidence": False,
            }
        )
        service = GroundedAnswerService(client=client)

        answer, citations = await service.answer("Is bicycle theft covered?", [search_result()])

        self.assertEqual(answer.evidence_ids, ["[E2]"])
        self.assertEqual(citations[0].evidence_id, "[E2]")
        self.assertEqual(citations[0].page_numbers, [22])
        self.assertTrue(citations[0].added_context)
        prompt = client.chat.completions.create.await_args.kwargs["messages"][1]["content"]
        self.assertIn("[E1]", prompt)
        self.assertIn("[E2]", prompt)

    async def test_rejects_evidence_id_not_supplied_by_backend(self):
        service = GroundedAnswerService(
            client=self.client(
                {
                    "answer": "Unsupported answer",
                    "evidence_ids": ["[E99]"],
                    "insufficient_evidence": False,
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "unknown evidence IDs"):
            await service.answer("Question", [search_result()])

    async def test_returns_insufficient_without_calling_llm_when_no_evidence(self):
        client = self.client({})
        answer, citations = await GroundedAnswerService(client=client).answer("Question", [])

        self.assertTrue(answer.insufficient_evidence)
        self.assertEqual(citations, [])
        client.chat.completions.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
