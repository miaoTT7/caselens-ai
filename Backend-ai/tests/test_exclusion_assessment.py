import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.claim_schemas import (
    Claim,
    ClaimFact,
    CoverageAssessment,
    EvidenceRef,
    ExclusionAssessmentRequest,
    Incident,
)
from api.exclusion_assessment import ExclusionAssessmentService
from api.schemas import SemanticSearchResult


class ExclusionAssessmentTests(unittest.IsolatedAsyncioTestCase):
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
    def result(document_id):
        return SemanticSearchResult(
            chunk_id=uuid.uuid4(),
            document_id=document_id,
            text="Theft is excluded when the bicycle was left unlocked.",
            score=0.92,
            chunk_type="text",
            source_file="policy-v1.pdf",
            source_format="pdf",
            page_numbers=[23],
            section_path=["F2 Bicycle theft", "Exclusions"],
            sheet=None,
            row_start=None,
            row_end=None,
            email_metadata={},
            rank=1,
        )

    @staticmethod
    def coverage(document_id):
        return CoverageAssessment(
            id=uuid.uuid4(),
            coverage_reference="F2 Bicycle theft",
            status="covered",
            policy_evidence=[
                EvidenceRef(
                    id="[P1]",
                    evidence_type="policy",
                    document_id=document_id,
                    source_file="policy-v1.pdf",
                    text_quote="Bicycle theft cover.",
                )
            ],
        )

    async def assess(self, *, condition_result, fact_value, evidence_id="P1"):
        document_id = uuid.uuid4()
        incident = Incident(id=uuid.uuid4(), incident_type="theft")
        claim = Claim(id=uuid.uuid4(), incidents=[incident])
        coverage = self.coverage(document_id)
        facts = []
        if fact_value is not None:
            facts.append(
                ClaimFact(
                    id=uuid.uuid4(),
                    fact_path="incidents[0].metadata.bicycle_locked",
                    value=fact_value,
                    status="confirmed",
                    claim_evidence=[
                        EvidenceRef(
                            id="[C1]",
                            evidence_type="claim",
                            source_file="fnol.txt",
                            text_quote=f"Bicycle locked: {fact_value}",
                        )
                    ],
                )
            )
        retrieval = SimpleNamespace(
            search=AsyncMock(return_value=[self.result(document_id)])
        )
        service = ExclusionAssessmentService(
            AsyncMock(),
            None,
            retrieval_service=retrieval,
            llm_client=self.client(
                {
                    "exclusions": [
                        {
                            "exclusion_reference": "Unlocked bicycle exclusion",
                            "condition_logic": "all",
                            "conditions": [
                                {
                                    "description": "The bicycle was left unlocked.",
                                    "result": condition_result,
                                    "fact_paths": ["incidents[0].metadata.bicycle_locked"],
                                    "policy_evidence_ids": [evidence_id],
                                }
                            ],
                        }
                    ]
                }
            ),
        )
        request = ExclusionAssessmentRequest(
            knowledge_base_id=uuid.uuid4(),
            claim=claim,
            facts=facts,
            coverage_assessments=[coverage],
            retrieval_limit=5,
        )
        return await service.assess(request), retrieval, coverage, incident

    async def test_explicitly_matched_exclusion_applies(self):
        result, retrieval, coverage, _ = await self.assess(
            condition_result="matched", fact_value=False
        )

        assessment = result.exclusion_assessments[0]
        self.assertEqual(assessment.status, "applies")
        self.assertEqual(assessment.coverage_assessment_id, coverage.id)
        self.assertEqual(assessment.policy_evidence[0].evidence_type, "policy")
        self.assertEqual(assessment.claim_evidence[0].id, "[C1]")
        search_request = retrieval.search.await_args.args[1]
        self.assertEqual(search_request.document_id, coverage.policy_evidence[0].document_id)

    async def test_explicitly_unmatched_exclusion_does_not_apply(self):
        result, _, _, _ = await self.assess(
            condition_result="unmatched", fact_value=True
        )

        self.assertEqual(result.exclusion_assessments[0].status, "does_not_apply")
        self.assertEqual(result.missing_information, [])

    async def test_missing_fact_is_indeterminate_and_creates_missing_information(self):
        result, _, _, incident = await self.assess(
            condition_result="unknown", fact_value=None
        )

        assessment = result.exclusion_assessments[0]
        self.assertEqual(assessment.status, "indeterminate")
        self.assertEqual(len(assessment.unknown_conditions), 1)
        self.assertEqual(len(assessment.missing_information_ids), 1)
        self.assertEqual(result.missing_information[0].related_incident_id, incident.id)

    async def test_invalid_policy_evidence_cannot_make_exclusion_apply(self):
        result, _, _, _ = await self.assess(
            condition_result="matched", fact_value=False, evidence_id="P99"
        )

        assessment = result.exclusion_assessments[0]
        self.assertEqual(assessment.status, "indeterminate")
        self.assertEqual(assessment.policy_evidence, [])
        self.assertEqual(len(result.missing_information), 1)

    async def test_absent_exclusion_conditions_remain_indeterminate(self):
        document_id = uuid.uuid4()
        coverage = self.coverage(document_id)
        claim = Claim(id=uuid.uuid4(), incidents=[Incident(id=uuid.uuid4())])
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.result(document_id)]))
        service = ExclusionAssessmentService(
            AsyncMock(),
            None,
            retrieval_service=retrieval,
            llm_client=self.client(
                {
                    "exclusions": [
                        {
                            "exclusion_reference": "Unresolved exclusion",
                            "condition_logic": "all",
                            "conditions": [],
                        }
                    ]
                }
            ),
        )

        result = await service.assess(
            ExclusionAssessmentRequest(
                knowledge_base_id=uuid.uuid4(),
                claim=claim,
                coverage_assessments=[coverage],
            )
        )

        self.assertEqual(result.exclusion_assessments[0].status, "indeterminate")
        self.assertEqual(len(result.missing_information), 1)


if __name__ == "__main__":
    unittest.main()
