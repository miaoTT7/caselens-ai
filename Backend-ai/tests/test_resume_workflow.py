import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError

from api.agent import CaseLensAgentState
from api.agent_state_persistence import UnsupportedAgentStateSchemaError
from api.claim_schemas import Claim, ClaimRecommendation, MissingInformation
from api.resume_workflow import (
    ClaimResumeService,
    ResumeSchemaVersionError,
    ResumeStateNotFoundError,
    ResumeStateValidationError,
)


def persisted(state, revision=3, schema_version=1):
    return SimpleNamespace(
        state=state,
        revision=revision,
        state_schema_version=schema_version,
    )


class ResumeWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.claim_id = uuid.uuid4()
        self.persistence = AsyncMock()
        self.service = ClaimResumeService(self.persistence)

    async def test_resume_ask_for_information_returns_blocking_questions(self):
        blocking = MissingInformation(
            id=uuid.uuid4(),
            field_path="incidents[0].location",
            reason="Location is required.",
            required_for="coverage_assessment",
            question="Where did the incident occur?",
            blocking=True,
        )
        non_blocking = MissingInformation(
            id=uuid.uuid4(),
            field_path="claim.currency",
            reason="Currency is useful.",
            required_for="claim_calculation",
            blocking=False,
        )
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase="coverage_assessment",
            missing_information=[blocking, non_blocking],
        )
        self.persistence.load_latest_state.return_value = persisted(state)

        result = await self.service.resume(self.claim_id)

        self.assertEqual(result.next_action, "ask_for_information")
        self.assertEqual([item.id for item in result.blocking_questions], [blocking.id])
        self.assertIsNone(result.target_phase)
        self.assertIsNone(result.next_tool_name)

    async def test_resume_rerun_returns_target_without_executing_it(self):
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase="exclusion_assessment",
            next_action="rerun_phase",
        )
        self.persistence.load_latest_state.return_value = persisted(state)

        result = await self.service.resume(self.claim_id)

        self.assertEqual(result.next_action, "rerun_phase")
        self.assertEqual(result.target_phase, "exclusion_assessment")
        self.assertIsNone(result.next_tool_name)

    async def test_resume_human_review_rebuilds_handoff(self):
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(),
            status="needs_human_review",
            reasons=["Manual review required."],
            human_review_required=True,
        )
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase="claim_recommendation",
            recommendation=recommendation,
        )
        self.persistence.load_latest_state.return_value = persisted(state)

        result = await self.service.resume(self.claim_id)

        self.assertEqual(result.next_action, "human_review")
        self.assertEqual(result.human_handoff.claim_id, self.claim_id)
        self.assertEqual(result.state.recommendation, recommendation)

    async def test_resume_completed_returns_persisted_final_state(self):
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(),
            status="recommend_approve",
            reasons=["Assessment complete."],
            human_review_required=False,
        )
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase="claim_recommendation",
            next_action="completed",
            completed_phases=["claim_recommendation"],
            recommendation=recommendation,
        )
        original = state.model_dump(mode="python")
        self.persistence.load_latest_state.return_value = persisted(state, revision=7)

        result = await self.service.resume(self.claim_id)

        self.assertEqual(result.next_action, "completed")
        self.assertEqual(result.revision, 7)
        self.assertEqual(result.state.recommendation.id, recommendation.id)
        self.assertEqual(state.model_dump(mode="python"), original)

    async def test_resume_continue_returns_next_tool_without_execution(self):
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase="coverage_assessment",
            next_action="continue_assessment",
        )
        self.persistence.load_latest_state.return_value = persisted(state)

        result = await self.service.resume(self.claim_id)

        self.assertEqual(result.next_action, "continue_assessment")
        self.assertEqual(result.next_tool_name, "assess_coverage")

    async def test_resume_state_not_found_is_explicit(self):
        self.persistence.load_latest_state.return_value = None

        with self.assertRaises(ResumeStateNotFoundError):
            await self.service.resume(self.claim_id)

    async def test_resume_schema_version_error_is_explicit(self):
        self.persistence.load_latest_state.side_effect = UnsupportedAgentStateSchemaError(
            "Unsupported agent state schema version: 99"
        )

        with self.assertRaises(ResumeSchemaVersionError) as raised:
            await self.service.resume(self.claim_id)

        self.assertIsInstance(
            raised.exception.__cause__, UnsupportedAgentStateSchemaError
        )

    async def test_resume_invalid_state_is_explicit(self):
        self.persistence.load_latest_state.return_value = persisted({
            "claim": {"id": str(self.claim_id)},
            "current_phase": "not_a_phase",
        })

        with self.assertRaises(ResumeStateValidationError) as raised:
            await self.service.resume(self.claim_id)

        self.assertIsInstance(raised.exception.__cause__, ValidationError)

    async def test_resume_claim_id_mismatch_is_invalid(self):
        state = CaseLensAgentState(claim=Claim(id=uuid.uuid4()))
        self.persistence.load_latest_state.return_value = persisted(state)

        with self.assertRaises(ResumeStateValidationError):
            await self.service.resume(self.claim_id)

    async def test_resume_has_no_provider_tool_or_orchestrator_side_effects(self):
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase="claim_calculation",
            next_action="continue_assessment",
        )
        self.persistence.load_latest_state.return_value = persisted(state)
        provider = AsyncMock()
        retrieval = AsyncMock()
        tool_registry = Mock()
        orchestrator = AsyncMock()

        result = await self.service.resume(self.claim_id)

        self.assertEqual(result.next_tool_name, "calculate_claim")
        provider.assert_not_called()
        retrieval.assert_not_called()
        tool_registry.assert_not_called()
        orchestrator.assert_not_called()
        self.persistence.load_latest_state.assert_awaited_once_with(self.claim_id)


if __name__ == "__main__":
    unittest.main()
