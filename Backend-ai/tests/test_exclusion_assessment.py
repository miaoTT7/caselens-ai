import json
import unittest
import uuid
import httpx
from pydantic import ValidationError
from types import SimpleNamespace
from unittest.mock import AsyncMock

from groq import BadRequestError

from api.claim_schemas import (
    Claim,
    ClaimFact,
    CoverageAssessment,
    EvidenceRef,
    ExclusionAssessmentRequest,
    Incident,
)
from api.exclusion_assessment import ExclusionAssessmentService, _ExclusionPayload
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

    async def test_bicycle_exclusion_retries_json_generation_failure_once(self):
        document_id = uuid.uuid4()
        coverage = self.coverage(document_id)
        claim = Claim(id=uuid.uuid4(), incidents=[Incident(
            id=uuid.uuid4(), incident_type="theft", location="outside Zurich station"
        )])
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.result(document_id)]))
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(400, request=request)
        provider_error = BadRequestError(
            "json_validate_failed", response=response,
            body={"error": {"code": "json_validate_failed"}},
        )
        successful = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "exclusions": [{
                "exclusion_reference": "Art. 110.1.3 Simple theft outside the home",
                "condition_logic": "all",
                "conditions": [{
                    "description": "The theft occurred outside the home.",
                    "result": "matched",
                    "fact_paths": ["incidents[0].location"],
                    "policy_evidence_ids": ["P1"],
                }],
            }]
        })))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
        client.chat.completions.create = AsyncMock(side_effect=[provider_error, successful])
        fact = ClaimFact(
            id=uuid.uuid4(), fact_path="incidents[0].location",
            value="outside Zurich station", status="confirmed",
        )
        service = ExclusionAssessmentService(
            AsyncMock(), None, retrieval_service=retrieval, llm_client=client
        )

        result = await service.assess(ExclusionAssessmentRequest(
            knowledge_base_id=uuid.uuid4(), claim=claim, facts=[fact],
            coverage_assessments=[coverage], retrieval_limit=8,
        ))

        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(result.exclusion_assessments[0].status, "applies")

    def test_captured_groq_shape_removes_only_empty_exclusion_entries(self):
        payload = _ExclusionPayload.model_validate({
            "exclusions": [
                {
                    "exclusion_reference": "P1",
                    "condition_logic": "all",
                    "conditions": [{
                        "description": (
                            "Theft is simple theft and occurs outside the insured premises "
                            "(outside Zurich station) and is not covered by supplementary insurance."
                        ),
                        "result": "matched",
                        "fact_paths": [
                            "incidents[0].incident_type",
                            "incidents[0].location",
                        ],
                        "policy_evidence_ids": ["P1"],
                    }],
                },
                "",
                {
                    "exclusion_reference": "P8-P9",
                    "condition_logic": "any",
                    "conditions": [{
                        "description": "Cash assets are excluded from coverage.",
                        "result": "unknown",
                        "fact_paths": ["claim.claimed_amount"],
                        "policy_evidence_ids": ["P8", "P9"],
                    }],
                },
            ]
        })

        self.assertEqual(len(payload.exclusions), 2)
        self.assertEqual(
            [item.exclusion_reference for item in payload.exclusions],
            ["P1", "P8-P9"],
        )

    def test_non_empty_invalid_exclusion_entry_still_fails_validation(self):
        with self.assertRaises(ValidationError) as context:
            _ExclusionPayload.model_validate({"exclusions": ["not-an-object"]})

        self.assertIn("exclusions.0", str(context.exception))


if __name__ == "__main__":
    unittest.main()
