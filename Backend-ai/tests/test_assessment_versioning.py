import unittest
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from api.agent import (
    CaseLensAgent,
    CaseLensAgentState,
    MissingInformationAnswer,
    MissingInformationAnswerRequest,
)
from api.agent_tools import ToolExecutionResult
from api.assessment_versioning import AssessmentVersionService
from api.claim_schemas import (
    Claim,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimRecommendation,
    CoverageAssessment,
    ExclusionAssessment,
    MissingInformation,
)
from api.models import AssessmentVersionRecord


class FakeAssessmentVersionRepository:
    def __init__(self):
        self.records: list[AssessmentVersionRecord] = []
        self.locked_claims: list[uuid.UUID] = []

    async def acquire_claim_lock(self, claim_id):
        self.locked_claims.append(claim_id)

    async def create_version(self, **values):
        record = AssessmentVersionRecord(
            assessment_version_id=uuid.uuid4(), **values
        )
        record.created_at = datetime.now(UTC)
        self.records.append(record)
        return record

    async def get_latest_version(self, claim_id):
        matches = [item for item in self.records if item.claim_id == claim_id]
        return max(matches, key=lambda item: item.version_number) if matches else None

    async def list_versions(self, claim_id):
        return sorted(
            (item for item in self.records if item.claim_id == claim_id),
            key=lambda item: item.version_number,
        )

    async def get_version(self, version_id):
        return next(
            (item for item in self.records if item.assessment_version_id == version_id),
            None,
        )


def stable_state(claim_id=None):
    claim_id = claim_id or uuid.uuid4()
    coverage = CoverageAssessment(
        id=uuid.uuid4(),
        coverage_reference="Bicycle theft",
        status="potentially_covered",
    )
    recommendation = ClaimRecommendation(
        id=uuid.uuid4(),
        status="needs_information",
        reasons=["More information is required."],
        human_review_required=False,
    )
    return CaseLensAgentState(
        claim=Claim(id=claim_id, description="Bicycle theft"),
        coverage_assessments=[coverage],
        recommendation=recommendation,
        completed_phases=["coverage_assessment", "claim_recommendation"],
        current_phase="claim_recommendation",
        next_action="ask_for_information",
    )


class AssessmentVersionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = FakeAssessmentVersionRepository()
        self.service = AssessmentVersionService(self.repository)

    async def test_v1_and_v2_are_immutable_separate_snapshots_with_parent_link(self):
        claim_id = uuid.uuid4()
        first_state = stable_state(claim_id)
        first_trigger = uuid.uuid4()

        v1 = await self.service.create_version(
            first_state,
            trigger_type="initial_assessment",
            trigger_reference_id=first_trigger,
        )
        second_state = first_state.model_copy(deep=True)
        second_state.exclusion_assessments = [ExclusionAssessment(
            id=uuid.uuid4(),
            coverage_assessment_id=second_state.coverage_assessments[0].id,
            exclusion_reference="Unlocked bicycle",
            status="does_not_apply",
        )]
        second_state.recommendation = ClaimRecommendation(
            id=uuid.uuid4(),
            status="recommend_approve",
            reasons=["Assessment complete."],
            human_review_required=False,
        )
        second_state.next_action = "completed"
        second_trigger = uuid.uuid4()

        v2 = await self.service.create_version(
            second_state,
            trigger_type="selective_rerun",
            trigger_reference_id=second_trigger,
            rerun_from_phase="exclusion_assessment",
        )

        self.assertEqual(v1.version_number, 1)
        self.assertIsNone(v1.parent_version_id)
        self.assertEqual(v1.trigger_reference_id, first_trigger)
        self.assertEqual(v2.version_number, 2)
        self.assertEqual(v2.parent_version_id, v1.assessment_version_id)
        self.assertEqual(v2.rerun_from_phase, "exclusion_assessment")
        self.assertEqual(v2.trigger_reference_id, second_trigger)
        self.assertEqual(v1.phase_outputs["exclusion_assessments"], [])
        self.assertEqual(
            v2.phase_outputs["coverage_assessments"],
            v1.phase_outputs["coverage_assessments"],
        )
        self.assertEqual(len(v2.phase_outputs["exclusion_assessments"]), 1)
        self.assertEqual(v1.recommendation["status"], "needs_information")
        self.assertEqual(v2.recommendation["status"], "recommend_approve")

    async def test_repository_read_methods_are_exposed_by_service(self):
        state = stable_state()
        created = await self.service.create_version(
            state, trigger_type="initial_assessment"
        )

        self.assertIs(
            await self.service.get_latest_version(state.claim.id), created
        )
        self.assertEqual(
            await self.service.list_versions(state.claim.id), [created]
        )
        self.assertIs(
            await self.service.get_version(created.assessment_version_id), created
        )

    async def test_unstable_or_fact_only_state_does_not_create_version(self):
        fact_only = CaseLensAgentState(claim=Claim(id=uuid.uuid4()))
        failed_rerun = stable_state().model_copy(update={"next_action": "rerun_phase"})

        self.assertFalse(self.service.is_stable_assessment(fact_only))
        self.assertFalse(self.service.is_stable_assessment(failed_rerun))
        with self.assertRaisesRegex(ValueError, "stable structured result"):
            await self.service.create_version(
                fact_only, trigger_type="initial_assessment"
            )
        self.assertEqual(self.repository.records, [])


class AssessmentVersionAgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_assessment_uses_fnol_event_as_v1_trigger(self):
        state = stable_state()
        response = ClaimAssessmentResponse(
            status="partial",
            completed_phases=state.completed_phases,
            claim=state.claim,
            coverage_assessments=state.coverage_assessments,
            recommendation=state.recommendation,
        )
        orchestrator = AsyncMock()
        orchestrator.assess.return_value = response
        fnol_event_id = uuid.uuid4()
        history = AsyncMock()
        history.record_initial_assessment.return_value = [SimpleNamespace(
            event_id=fnol_event_id, event_type="fnol_submitted"
        )]
        version_service = Mock()
        version_service.is_stable_assessment.return_value = True
        version_service.create_version = AsyncMock()
        agent = CaseLensAgent(
            orchestrator,
            history_service=history,
            assessment_version_service=version_service,
        )

        await agent.assess(ClaimAssessmentRequest(
            fnol_text="My bicycle was stolen.",
            knowledge_base_id=uuid.uuid4(),
        ))

        kwargs = version_service.create_version.await_args.kwargs
        self.assertEqual(kwargs["trigger_type"], "initial_assessment")
        self.assertEqual(kwargs["trigger_reference_id"], fnol_event_id)

    async def test_user_answer_alone_does_not_create_version(self):
        missing = MissingInformation(
            id=uuid.uuid4(),
            field_path="incidents[0].location",
            reason="Location required.",
            required_for="coverage_assessment",
            blocking=True,
        )
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            missing_information=[missing],
            next_action="ask_for_information",
        )
        version_service = Mock()
        version_service.create_version = AsyncMock()
        agent = CaseLensAgent(AsyncMock(), assessment_version_service=version_service)

        await agent.apply_missing_information_answers(
            state,
            MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
                missing_information_id=missing.id,
                value="Zurich",
            )]),
        )

        version_service.create_version.assert_not_awaited()

    async def test_successful_rerun_creates_v2_with_completed_event_reference(self):
        before = stable_state()
        before.current_phase = "exclusion_assessment"
        before.next_action = "rerun_phase"
        after = stable_state(before.claim.id)
        after.next_action = "completed"
        completion_event_id = uuid.uuid4()
        history = AsyncMock()
        history.record_rerun_started.return_value = SimpleNamespace(event_id=uuid.uuid4())
        history.record_rerun_completed.return_value = [SimpleNamespace(
            event_id=completion_event_id,
            event_type="selective_rerun_completed",
        )]
        version_service = Mock()
        version_service.is_stable_assessment.return_value = True
        version_service.create_version = AsyncMock()
        tool = AsyncMock()
        tool.execute.return_value = ToolExecutionResult(
            tool_name="selective_rerun", success=True, output=after
        )
        registry = Mock()
        registry.get.return_value = tool
        agent = CaseLensAgent(
            AsyncMock(),
            tool_registry=registry,
            history_service=history,
            assessment_version_service=version_service,
        )

        await agent.execute_next(before)

        kwargs = version_service.create_version.await_args.kwargs
        self.assertEqual(kwargs["trigger_type"], "selective_rerun")
        self.assertEqual(kwargs["trigger_reference_id"], completion_event_id)
        self.assertEqual(kwargs["rerun_from_phase"], "exclusion_assessment")

    async def test_failed_rerun_does_not_create_version(self):
        before = stable_state()
        before.current_phase = "exclusion_assessment"
        before.next_action = "rerun_phase"
        failed = before.model_copy(deep=True)
        failed.rerun_failure_count = 1
        tool = AsyncMock()
        tool.execute.return_value = ToolExecutionResult(
            tool_name="selective_rerun", success=True, output=failed
        )
        registry = Mock()
        registry.get.return_value = tool
        version_service = Mock()
        version_service.is_stable_assessment.return_value = True
        version_service.create_version = AsyncMock()
        agent = CaseLensAgent(
            AsyncMock(),
            tool_registry=registry,
            assessment_version_service=version_service,
        )

        await agent.execute_next(before)

        version_service.create_version.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
