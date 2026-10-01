import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from groq import BadRequestError

from api.claim_schemas import (
    Claim,
    ClaimFact,
    CoverageAssessment,
    EvidenceRef,
    Incident,
    ObligationAssessmentRequest,
)
from api.obligation_assessment import ObligationAssessmentService
from api.schemas import SemanticSearchResult


class ObligationAssessmentTests(unittest.IsolatedAsyncioTestCase):
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
            text=(
                "The policyholder must report theft to police without delay. A culpable breach that "
                "affects the loss may permit a reduction proportionate to that effect."
            ),
            score=0.93,
            chunk_type="text",
            source_file="policy-v1.pdf",
            source_format="pdf",
            page_numbers=[31],
            section_path=["Claims duties", "Police reporting"],
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

    @staticmethod
    def fact(path, value, evidence_id):
        return ClaimFact(
            id=uuid.uuid4(),
            fact_path=path,
            value=value,
            status="confirmed",
            claim_evidence=[
                EvidenceRef(
                    id=evidence_id,
                    evidence_type="claim",
                    source_file="fnol.txt",
                    text_quote=f"{path}: {value}",
                )
            ],
        )

    async def run_assessment(self, payload, facts):
        document_id = uuid.uuid4()
        incident = Incident(id=uuid.uuid4(), incident_type="theft")
        coverage = self.coverage(document_id)
        retrieval = SimpleNamespace(
            search=AsyncMock(return_value=[self.result(document_id)])
        )
        service = ObligationAssessmentService(
            AsyncMock(), None,
            retrieval_service=retrieval,
            llm_client=self.client({"obligations": [payload]}),
        )
        result = await service.assess(
            ObligationAssessmentRequest(
                knowledge_base_id=uuid.uuid4(),
                claim=Claim(id=uuid.uuid4(), incidents=[incident]),
                facts=facts,
                coverage_assessments=[coverage],
            )
        )
        return result, retrieval, coverage, incident

    async def test_all_grounded_conditions_satisfied(self):
        path = "incidents[0].metadata.reported_to_police"
        result, retrieval, coverage, _ = await self.run_assessment(
            {
                "obligation_reference": "Report theft to police",
                "conditions": [{
                    "description": "The theft was reported to police without delay.",
                    "result": "matched",
                    "fact_paths": [path],
                    "policy_evidence_ids": ["P1"],
                }],
            },
            [self.fact(path, True, "[C1]")],
        )

        assessment = result.obligation_assessments[0]
        self.assertEqual(assessment.status, "satisfied")
        self.assertEqual(assessment.coverage_assessment_id, coverage.id)
        self.assertIsNone(assessment.culpable_breach)
        self.assertEqual(assessment.effect_on_loss, "unknown")
        self.assertIsNone(assessment.permitted_consequence)
        self.assertEqual(retrieval.search.await_args.args[1].document_id,
                         coverage.policy_evidence[0].document_id)

    async def test_breach_keeps_culpability_effect_and_consequence_separate(self):
        report_path = "incidents[0].metadata.reported_to_police"
        culpability_path = "incidents[0].metadata.late_report_negligent"
        effect_path = "incidents[0].metadata.late_report_affected_loss"
        result, _, _, _ = await self.run_assessment(
            {
                "obligation_reference": "Report theft to police",
                "conditions": [{
                    "description": "The theft was reported to police without delay.",
                    "result": "unmatched",
                    "fact_paths": [report_path],
                    "policy_evidence_ids": ["P1"],
                }],
                "requires_culpable_breach": True,
                "culpable_breach": True,
                "culpability_fact_paths": [culpability_path],
                "culpability_policy_evidence_ids": ["P1"],
                "requires_effect_on_loss": True,
                "effect_on_loss": "affected",
                "effect_fact_paths": [effect_path],
                "effect_policy_evidence_ids": ["P1"],
                "permitted_consequence": "A proportionate reduction may be permitted.",
                "consequence_policy_evidence_ids": ["P1"],
            },
            [
                self.fact(report_path, False, "[C1]"),
                self.fact(culpability_path, True, "[C2]"),
                self.fact(effect_path, True, "[C3]"),
            ],
        )

        assessment = result.obligation_assessments[0]
        self.assertEqual(assessment.status, "breached")
        self.assertIs(assessment.culpable_breach, True)
        self.assertEqual(assessment.effect_on_loss, "affected")
        self.assertEqual(
            assessment.permitted_consequence,
            "A proportionate reduction may be permitted.",
        )

    async def test_missing_required_fact_is_indeterminate(self):
        path = "incidents[0].metadata.reported_to_police"
        result, _, _, incident = await self.run_assessment(
            {
                "obligation_reference": "Report theft to police",
                "conditions": [{
                    "description": "The theft was reported to police without delay.",
                    "result": "unknown",
                    "fact_paths": [path],
                    "policy_evidence_ids": ["P1"],
                }],
            },
            [],
        )

        assessment = result.obligation_assessments[0]
        self.assertEqual(assessment.status, "indeterminate")
        self.assertEqual(len(assessment.missing_information_ids), 1)
        self.assertEqual(result.missing_information[0].field_path, path)
        self.assertEqual(result.missing_information[0].related_incident_id, incident.id)

    async def test_bicycle_theft_groq_null_effect_is_treated_as_unknown(self):
        facts = [
            self.fact("incidents[0].incident_type", "theft", "[C1]"),
            self.fact("incidents[0].event_date", "2026-08-12", "[C2]"),
            self.fact("incidents[0].location", "outside Zurich station", "[C3]"),
            self.fact("claim.claimed_amount", "1800", "[C4]"),
            self.fact("claim.currency", "CHF", "[C5]"),
        ]
        result, _, _, _ = await self.run_assessment(
            {
                "obligation_reference": "Art. 6.1",
                "conditions": [{
                    "description": "Notify police immediately upon theft",
                    "result": "unknown",
                    "fact_paths": ["incidents[0].incident_type"],
                    "policy_evidence_ids": ["P1"],
                }],
                "requires_culpable_breach": True,
                "culpable_breach": None,
                "culpability_fact_paths": [],
                "culpability_policy_evidence_ids": ["P1"],
                "requires_effect_on_loss": True,
                "effect_on_loss": None,
                "effect_fact_paths": [],
                "effect_policy_evidence_ids": ["P1"],
                "permitted_consequence": "rejection or reduction",
                "consequence_policy_evidence_ids": ["P1"],
            },
            facts,
        )

        assessment = result.obligation_assessments[0]
        self.assertEqual(assessment.status, "indeterminate")
        self.assertEqual(assessment.effect_on_loss, "unknown")
        self.assertEqual(assessment.obligation_reference, "Art. 6.1")

    async def test_bicycle_obligation_retries_json_generation_failure_once(self):
        document_id = uuid.uuid4()
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.result(document_id)]))
        client = self.client({
            "obligations": [{
                "obligation_reference": "Art. 6.1",
                "conditions": [{
                    "description": "Notify police immediately upon theft",
                    "result": "unknown",
                    "fact_paths": ["incidents[0].metadata.reported_to_police"],
                    "policy_evidence_ids": ["P1"],
                }],
                "effect_on_loss": "unknown",
            }]
        })
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(400, request=request)
        client.chat.completions.create.side_effect = [
            BadRequestError(
                "json_validate_failed",
                response=response,
                body={"error": {"code": "json_validate_failed"}},
            ),
            client.chat.completions.create.return_value,
        ]
        service = ObligationAssessmentService(
            AsyncMock(), None, retrieval_service=retrieval, llm_client=client
        )
        result = await service.assess(
            ObligationAssessmentRequest(
                knowledge_base_id=uuid.uuid4(),
                claim=Claim(
                    id=uuid.uuid4(),
                    incidents=[Incident(id=uuid.uuid4(), incident_type="theft")],
                ),
                facts=[self.fact("incidents[0].incident_type", "theft", "[C1]")],
                coverage_assessments=[self.coverage(document_id)],
            )
        )

        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(result.obligation_assessments[0].obligation_reference, "Art. 6.1")

    async def test_both_json_generation_attempts_fail_returns_unavailable(self):
        document_id = uuid.uuid4()
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.result(document_id)]))
        client = self.client({})
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        response = httpx.Response(400, request=request)
        errors = [
            BadRequestError(
                "json_validate_failed",
                response=response,
                body={"error": {"code": "json_validate_failed"}},
            )
            for _ in range(2)
        ]
        client.chat.completions.create.side_effect = errors
        service = ObligationAssessmentService(
            AsyncMock(), None, retrieval_service=retrieval, llm_client=client
        )

        result = await service.assess(ObligationAssessmentRequest(
            knowledge_base_id=uuid.uuid4(),
            claim=Claim(
                id=uuid.uuid4(),
                incidents=[Incident(id=uuid.uuid4(), incident_type="theft")],
            ),
            facts=[self.fact("incidents[0].incident_type", "theft", "[C1]")],
            coverage_assessments=[self.coverage(document_id)],
        ))

        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.obligation_assessments, [])
        self.assertTrue(result.missing_information[0].blocking)
        self.assertTrue(result.missing_information[0].metadata["assessment_unavailable"])

    async def test_invalid_policy_evidence_cannot_prove_breach(self):
        path = "incidents[0].metadata.reported_to_police"
        result, _, _, _ = await self.run_assessment(
            {
                "obligation_reference": "Report theft to police",
                "conditions": [{
                    "description": "The theft was reported to police without delay.",
                    "result": "unmatched",
                    "fact_paths": [path],
                    "policy_evidence_ids": ["P99"],
                }],
            },
            [self.fact(path, False, "[C1]")],
        )

        self.assertEqual(result.obligation_assessments[0].status, "indeterminate")
        self.assertEqual(result.obligation_assessments[0].policy_evidence, [])


if __name__ == "__main__":
    unittest.main()
