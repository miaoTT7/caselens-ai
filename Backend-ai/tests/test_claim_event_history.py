import unittest
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from api.agent import (
    CaseLensAgent,
    CaseLensAgentState,
    MissingInformationAnswer,
    MissingInformationAnswerRequest,
)
from api.agent_tools import ToolExecutionResult
from api.claim_event_history import (
    ClaimEventCreate,
    ClaimEventHistoryService,
    ClaimEventPersistenceError,
)
from api.claim_schemas import (
    Claim,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimFact,
    ClaimRecommendation,
    MissingInformation,
)
from api.models import ClaimEventRecord


class FakeClaimEventRepository:
    def __init__(self):
        self.records: list[ClaimEventRecord] = []
        self.failure: Exception | None = None

    async def append_events(self, values):
        if self.failure is not None:
            raise self.failure
        now = datetime.now(UTC)
        records = []
        for value in values:
            record = ClaimEventRecord(event_id=uuid.uuid4(), **value)
            record.created_at = now
            records.append(record)
        self.records.extend(records)
        return records

    async def list_events(self, claim_id):
        return [item for item in self.records if item.claim_id == claim_id]

    async def list_fact_history(self, claim_id, field_path):
        return [
            item for item in self.records
            if item.claim_id == claim_id
            and item.field_path == field_path
            and item.event_type in {
                "claim_fact_extracted", "claim_fact_updated", "user_answer_applied"
            }
        ]


class ClaimEventHistoryServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = FakeClaimEventRepository()
        self.service = ClaimEventHistoryService(self.repository)

    async def test_append_and_list_event(self):
        claim_id = uuid.uuid4()
        await self.service.append_event(ClaimEventCreate(
            claim_id=claim_id,
            event_type="fnol_submitted",
            actor_type="user",
            raw_answer="My bicycle was stolen.",
        ))

        events = await self.service.list_events(claim_id)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "fnol_submitted")
        self.assertEqual(events[0].raw_answer, "My bicycle was stolen.")

    async def test_answer_keeps_raw_and_normalized_values_separate(self):
        claim_id = uuid.uuid4()
        missing = MissingInformation(
            id=uuid.uuid4(),
            field_path="claim.claimed_amount",
            reason="Amount required.",
            required_for="claim_calculation",
            blocking=True,
        )
        before = CaseLensAgentState(
            claim=Claim(id=claim_id),
            missing_information=[missing],
            next_action="ask_for_information",
        )
        answer_request = MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
            missing_information_id=missing.id,
            value=Decimal("1800.00"),
            original_text="CHF 1,800",
        )])
        from api.agent import MissingInformationInteraction
        after = MissingInformationInteraction.apply_answers(before, answer_request)

        await self.service.record_answers(
            before=before,
            after=after,
            request=answer_request,
            assessment_session_id=uuid.uuid4(),
        )

        received, applied, changed = self.repository.records
        self.assertEqual(received.raw_answer, "CHF 1,800")
        self.assertIsNone(received.normalized_value)
        self.assertEqual(applied.raw_answer, "CHF 1,800")
        self.assertEqual(applied.normalized_value, "1800.00")
        self.assertEqual(changed.old_value, None)
        self.assertEqual(changed.new_value, "1800.00")
        history = await self.service.list_fact_history(claim_id, "claim.claimed_amount")
        self.assertEqual([item.event_type for item in history], [
            "user_answer_applied", "claim_fact_updated"
        ])

    async def test_persistence_failure_is_explicit(self):
        self.repository.failure = RuntimeError("database unavailable")

        with self.assertRaises(ClaimEventPersistenceError) as raised:
            await self.service.append_event(ClaimEventCreate(
                claim_id=uuid.uuid4(),
                event_type="phase_completed",
                actor_type="system",
                related_phase="fact_validation",
            ))

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    async def test_initial_assessment_records_existing_outputs_only(self):
        claim_id = uuid.uuid4()
        state = CaseLensAgentState(
            claim=Claim(id=claim_id),
            facts=[ClaimFact(
                id=uuid.uuid4(),
                fact_path="incidents[0].incident_type",
                value="theft",
                status="extracted",
                confidence=0.9,
            )],
            completed_phases=["claim_facts_extraction", "fact_validation"],
        )

        await self.service.record_initial_assessment(
            fnol_text="My bicycle was stolen.",
            state=state,
            assessment_session_id=None,
        )

        self.assertEqual([item.event_type for item in self.repository.records], [
            "fnol_submitted",
            "claim_fact_extracted",
            "phase_completed",
            "phase_completed",
        ])


class ClaimEventAgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_emits_answer_history_at_existing_mapping_boundary(self):
        repository = FakeClaimEventRepository()
        history = ClaimEventHistoryService(repository)
        missing = MissingInformation(
            id=uuid.uuid4(),
            field_path="incidents[0].location",
            reason="Location required.",
            required_for="exclusion_assessment",
            blocking=True,
        )
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            missing_information=[missing],
            next_action="ask_for_information",
        )
        agent = CaseLensAgent(AsyncMock(), history_service=history)

        updated = await agent.apply_missing_information_answers(
            state,
            MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
                missing_information_id=missing.id,
                value="outside Zurich station",
            )]),
        )

        self.assertEqual(updated.facts[0].value, "outside Zurich station")
        self.assertEqual(updated.next_action, "rerun_phase")
        self.assertEqual(len(repository.records), 3)

    async def test_agent_emits_rerun_start_and_completion(self):
        repository = FakeClaimEventRepository()
        history = ClaimEventHistoryService(repository)
        output = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            current_phase="claim_recommendation",
            next_action="completed",
        )
        state = output.model_copy(update={
            "current_phase": "coverage_assessment",
            "next_action": "rerun_phase",
        })
        tool = AsyncMock()
        tool.execute.return_value = ToolExecutionResult(
            tool_name="selective_rerun", success=True, output=output
        )
        registry = Mock()
        registry.get.return_value = tool
        agent = CaseLensAgent(
            AsyncMock(), tool_registry=registry, history_service=history
        )

        await agent.execute_next(state)

        self.assertEqual([item.event_type for item in repository.records], [
            "selective_rerun_started", "selective_rerun_completed"
        ])

    async def test_agent_emits_human_handoff_history(self):
        repository = FakeClaimEventRepository()
        history = ClaimEventHistoryService(repository)
        state = CaseLensAgentState(
            claim=Claim(id=uuid.uuid4()),
            current_phase="claim_recommendation",
            next_action="human_review",
            recommendation=ClaimRecommendation(
                id=uuid.uuid4(),
                status="needs_human_review",
                reasons=["Manual review required."],
                human_review_required=True,
            ),
        )
        agent = CaseLensAgent(AsyncMock(), history_service=history)

        handoff = await agent.execute_next(state)

        self.assertEqual(handoff.claim_id, state.claim.id)
        self.assertEqual(len(repository.records), 1)
        self.assertEqual(repository.records[0].event_type, "human_handoff_created")
        self.assertEqual(repository.records[0].new_value["reason"], handoff.reason)

    async def test_agent_history_failure_is_not_silenced(self):
        repository = FakeClaimEventRepository()
        repository.failure = RuntimeError("write failed")
        history = ClaimEventHistoryService(repository)
        response = ClaimAssessmentResponse(
            status="completed",
            completed_phases=["claim_recommendation"],
            claim=Claim(id=uuid.uuid4()),
            recommendation=ClaimRecommendation(
                id=uuid.uuid4(),
                status="recommend_approve",
                reasons=["Complete."],
                human_review_required=False,
            ),
        )
        orchestrator = AsyncMock()
        orchestrator.assess.return_value = response
        agent = CaseLensAgent(orchestrator, history_service=history)

        with self.assertRaises(ClaimEventPersistenceError):
            await agent.assess(ClaimAssessmentRequest(
                fnol_text="My bicycle was stolen.",
                knowledge_base_id=uuid.uuid4(),
            ))


if __name__ == "__main__":
    unittest.main()
