import unittest
import uuid
from unittest.mock import AsyncMock, Mock

from api.agent import CaseLensAgent, CaseLensAgentState
from api.claim_schemas import (
    Claim,
    ClaimRecommendation,
    CoverageAssessment,
    EvidenceRef,
    MissingInformation,
    ObligationAssessment,
)
from api.human_handoff import HumanReviewHandoff, HumanReviewHandoffBuilder


class HumanReviewHandoffBuilderTests(unittest.TestCase):
    def test_provider_unavailable_handoff_reuses_existing_evidence(self):
        evidence = EvidenceRef(
            id="[P1]",
            evidence_type="policy",
            document_id=uuid.uuid4(),
            chunk_id=uuid.uuid4(),
            source_file="policy.pdf",
            text_quote="The policyholder must notify the police.",
        )
        missing = MissingInformation(
            id=uuid.uuid4(),
            field_path="obligations.provider_assessment",
            reason="Provider unavailable.",
            required_for="obligation_assessment",
            blocking=True,
            metadata={"assessment_unavailable": True},
        )
        coverage = CoverageAssessment(
            id=uuid.uuid4(),
            coverage_reference="Theft",
            status="covered",
            policy_evidence=[evidence],
        )
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            completed_phases=["coverage_assessment", "exclusion_assessment"],
            current_phase="obligation_assessment",
            next_action="human_review",
            missing_information=[missing],
            coverage_assessments=[coverage],
            provider_errors=["LLMProviderError: Groq request failed"],
        )

        handoff = HumanReviewHandoffBuilder().build(state)

        self.assertEqual(handoff.reason, "provider_unavailable")
        self.assertEqual(handoff.blocking_missing_information[0].id, missing.id)
        self.assertIs(handoff.relevant_evidence[0], evidence)
        self.assertEqual(handoff.relevant_evidence[0].text_quote, evidence.text_quote)
        self.assertIn("obligations.provider_assessment", handoff.suggested_review_focus[1])

    def test_obligation_breach_reason_is_deterministic(self):
        coverage_id = uuid.uuid4()
        obligation = ObligationAssessment(
            id=uuid.uuid4(),
            coverage_assessment_id=coverage_id,
            obligation_reference="Notify police",
            status="breached",
            effect_on_loss="unknown",
        )
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(),
            status="needs_human_review",
            reasons=["A breach requires interpretation."],
            human_review_required=True,
            disclaimer="Decision support only.",
        )
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            next_action="human_review",
            obligation_assessments=[obligation],
            recommendation=recommendation,
        )

        handoff = HumanReviewHandoffBuilder().build(state)

        self.assertEqual(handoff.reason, "obligation_breach_requires_review")
        self.assertIn("Notify police", handoff.suggested_review_focus[-1])

    def test_repeated_failure_has_priority(self):
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            next_action="human_review",
            rerun_failure_count=2,
            provider_errors=["LLMProviderError: failed"],
        )

        self.assertEqual(
            HumanReviewHandoffBuilder().build(state).reason,
            "repeated_rerun_failure",
        )


class HumanReviewAgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_human_review_returns_handoff_without_executing_tool(self):
        registry = Mock()
        agent = CaseLensAgent(AsyncMock(), tool_registry=registry)
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            current_phase="claim_recommendation",
            next_action="human_review",
            recommendation=ClaimRecommendation(
                id=uuid.uuid4(),
                status="needs_human_review",
                reasons=["Structured assessments remain ambiguous."],
                human_review_required=True,
                disclaimer="Decision support only.",
            ),
        )

        result = await agent.execute_next(state)

        self.assertIsInstance(result, HumanReviewHandoff)
        registry.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
