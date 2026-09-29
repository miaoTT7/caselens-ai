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
