import json
import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.claim_schemas import (
    Claim,
    ClaimFact,
    CoverageAssessmentRequest,
    EvidenceRef,
    Incident,
    PolicyReference,
)
from api.coverage_assessment import CoverageAssessmentService
from api.schemas import SemanticSearchResult


class CoverageAssessmentTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def client(payload):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )
        )
        return client

    @staticmethod
    def policy_result(document_id):
        return SemanticSearchResult(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            text="Policy V1. Bicycle theft is covered when the bicycle is secured with a lock.",
            score=0.9,
            chunk_type="text",
            source_file="policy-v1.pdf",
            source_format="pdf",
            page_numbers=[22],
            section_path=["F2 Bicycle theft"],
            sheet=None,
            row_start=None,
            row_end=None,
            email_metadata={},
            rank=1,
        )

    async def test_identified_policy_and_matched_triggers_produce_covered(self):
        document_id = uuid.uuid4()
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.policy_result(document_id)]))
        claim_evidence = EvidenceRef(
            id="[C1]", evidence_type="claim", source_file="fnol.txt", text_quote="The bicycle was locked."
        )
        fact = ClaimFact(
            id=uuid.uuid4(),
            fact_path="incidents[0].metadata.bicycle_locked",
            value=True,
            status="extracted",
            confidence=0.9,
            claim_evidence=[claim_evidence],
        )
        claim = Claim(
            id=uuid.uuid4(),
            policy=PolicyReference(id=uuid.uuid4(), policy_version="V1"),
            incidents=[
                Incident(
                    id=uuid.uuid4(), incident_type="theft",
                    event_date=datetime(2026, 8, 12, tzinfo=timezone.utc), location="Zurich",
                )
            ],
        )
        service = CoverageAssessmentService(
            AsyncMock(), None, retrieval_service=retrieval,
            llm_client=self.client(
                {
                    "candidates": [{
                        "coverage_reference": "F2 Bicycle theft",
                        "conditions": [{
                            "description": "The bicycle was secured with a lock.",
                            "result": "matched",
                            "fact_paths": ["incidents[0].metadata.bicycle_locked"],
                            "policy_evidence_ids": ["P1"],
                        }],
                    }]
                }
            ),
        )

        result = await service.assess(
            CoverageAssessmentRequest(
                knowledge_base_id=uuid.uuid4(), claim=claim, facts=[fact], retrieval_limit=5
            )
        )

        self.assertEqual(result.applicable_policy.status, "identified")
        self.assertEqual(result.coverage_assessments[0].status, "covered")
        self.assertIsNone(result.coverage_assessments[0].exposure_id)
        self.assertEqual(result.coverage_assessments[0].policy_evidence[0].evidence_type, "policy")
        self.assertEqual(result.coverage_assessments[0].claim_evidence[0].id, "[C1]")

    async def test_missing_trigger_fact_produces_potential_and_missing_information(self):
        document_id = uuid.uuid4()
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.policy_result(document_id)]))
        claim = Claim(
            id=uuid.uuid4(),
            policy=PolicyReference(id=uuid.uuid4(), policy_version="V1"),
            incidents=[Incident(id=uuid.uuid4(), incident_type="theft")],
        )
        service = CoverageAssessmentService(
            AsyncMock(), None, retrieval_service=retrieval,
            llm_client=self.client(
                {
                    "candidates": [{
                        "coverage_reference": "F2 Bicycle theft",
                        "conditions": [
                            {
                                "description": "The event was theft.",
                                "result": "matched",
                                "fact_paths": ["incidents[0].incident_type"],
                                "policy_evidence_ids": ["[P1]"],
                            },
                            {
                                "description": "The bicycle was secured with a lock.",
                                "result": "unknown",
                                "fact_paths": ["incidents[0].metadata.bicycle_locked"],
                                "policy_evidence_ids": ["[P1]"],
                            },
                        ],
                    }]
                }
            ),
        )
        incident_fact = ClaimFact(
            id=uuid.uuid4(), fact_path="incidents[0].incident_type",
            value="theft", status="confirmed",
        )

        result = await service.assess(
            CoverageAssessmentRequest(
                knowledge_base_id=uuid.uuid4(), claim=claim, facts=[incident_fact]
            )
        )

        assessment = result.coverage_assessments[0]
        self.assertEqual(assessment.status, "potentially_covered")
        self.assertEqual(len(assessment.unknown_conditions), 1)
        self.assertEqual(len(assessment.missing_information_ids), 1)


if __name__ == "__main__":
    unittest.main()
