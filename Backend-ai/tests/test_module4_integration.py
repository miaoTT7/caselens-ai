"""Real-PostgreSQL, rollback-isolated integration test for Module 4."""

from __future__ import annotations

import json
import unittest
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

from dotenv import load_dotenv
from sqlalchemy import delete, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from api.agent import (
    CaseLensAgent,
    CaseLensAgentState,
    MissingInformationAnswer,
    MissingInformationAnswerRequest,
)
from api.agent_state_persistence import ClaimAgentStatePersistenceService
from api.agent_tools import ServiceTool, ToolRegistry
from api.assessment_versioning import AssessmentVersionService
from api.claim_event_history import ClaimEventHistoryService
from api.claim_schemas import (
    Claim,
    ClaimFact,
    ClaimRecommendation,
    CoverageAssessment,
    ExclusionAssessment,
    Incident,
    MissingInformation,
)
from api.context_retrieval import ContextRetrievalService
from api.database import get_engine
from api.models import AssessmentVersionRecord, ClaimEventRecord
from api.repositories import (
    AgentStateRevisionConflictError,
    AssessmentVersionRepository,
    ClaimAgentStateRepository,
    ClaimEventRepository,
)
from api.resume_workflow import ClaimResumeService


load_dotenv()


class ControlledSelectiveRerunner:
    """Returns a controlled downstream result and never calls Phase 1-7."""

    def __init__(self, output: CaseLensAgentState):
        self.output = output
        self.calls = 0

    async def rerun(self, state: CaseLensAgentState) -> CaseLensAgentState:
        self.calls += 1
        if state.current_phase != "exclusion_assessment":
            raise AssertionError("Unexpected selective-rerun target")
        return self.output.model_copy(deep=True)


class Module4PostgresIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.connection = await get_engine().connect()
        self.transaction = await self.connection.begin()
        self.session = AsyncSession(
            bind=self.connection,
            expire_on_commit=False,
            autoflush=False,
        )

        self.state_repository = ClaimAgentStateRepository(self.session)
        self.event_repository = ClaimEventRepository(self.session)
        self.version_repository = AssessmentVersionRepository(self.session)
        self.persistence = ClaimAgentStatePersistenceService(self.state_repository)
        self.history = ClaimEventHistoryService(self.event_repository)
        self.versions = AssessmentVersionService(self.version_repository)

    async def asyncTearDown(self):
        await self.session.close()
        if self.transaction.is_active:
            await self.transaction.rollback()
        await self.connection.close()

    @staticmethod
    def _initial_state() -> tuple[CaseLensAgentState, MissingInformation]:
        claim_id = uuid.uuid4()
        incident_id = uuid.uuid4()
        missing = MissingInformation(
            id=uuid.uuid4(),
            field_path="incidents[0].metadata.bicycle_locked",
            reason="The bicycle lock status is required to assess the theft exclusion.",
            required_for="exclusion_assessment",
            question="Was the bicycle locked when it was stolen?",
            related_incident_id=incident_id,
            blocking=True,
        )
        claim = Claim(
            id=claim_id,
            description="Bicycle theft outside Zurich station.",
            incidents=[Incident(
                id=incident_id,
                incident_type="theft",
                event_date=datetime(2026, 8, 12, tzinfo=UTC),
                location="outside Zurich station",
            )],
        )
        facts = [
            ClaimFact(id=uuid.uuid4(), fact_path="incidents[0].incident_type", value="theft", status="confirmed"),
            ClaimFact(id=uuid.uuid4(), fact_path="incidents[0].event_date", value="2026-08-12", status="confirmed"),
            ClaimFact(id=uuid.uuid4(), fact_path="incidents[0].location", value="outside Zurich station", status="confirmed"),
            ClaimFact(id=uuid.uuid4(), fact_path="claim.claimed_amount", value=Decimal("1800"), unit="CHF", status="confirmed"),
            ClaimFact(id=uuid.uuid4(), fact_path="claim.currency", value="CHF", status="confirmed"),
        ]
        coverage = CoverageAssessment(
            id=uuid.uuid4(),
            coverage_reference="Art. 110 Theft",
            status="potentially_covered",
            unknown_conditions=["Whether the bicycle was secured as required."],
            missing_information_ids=[missing.id],
        )
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(),
            status="needs_information",
            reasons=["Lock status is required."],
            human_review_required=False,
        )
        return CaseLensAgentState(
            claim=claim,
            facts=facts,
            missing_information=[missing],
            coverage_assessments=[coverage],
            recommendation=recommendation,
            completed_phases=[
                "claim_facts_extraction",
                "fact_validation",
                "coverage_assessment",
                "claim_recommendation",
            ],
            current_phase="exclusion_assessment",
            next_action="ask_for_information",
        ), missing

    @staticmethod
    def _completed_state(answered: CaseLensAgentState) -> CaseLensAgentState:
        coverage = answered.coverage_assessments[0]
        exclusion = ExclusionAssessment(
            id=uuid.uuid4(),
            coverage_assessment_id=coverage.id,
            exclusion_reference="Unsecured bicycle",
            status="does_not_apply",
            unmatched_conditions=["The bicycle was reported as locked."],
            claim_evidence=[answered.facts[-1].claim_evidence[0]],
        )
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(),
            status="recommend_approve",
            recommended_payable_amount=Decimal("1800"),
            currency="CHF",
            reasons=["The controlled rerun completed with no applicable exclusion."],
            supporting_assessment_ids=[coverage.id, exclusion.id],
            human_review_required=False,
        )
        return answered.model_copy(deep=True, update={
            "missing_information": [],
            "exclusion_assessments": [exclusion],
            "recommendation": recommendation,
            "completed_phases": [
                "claim_facts_extraction",
                "fact_validation",
                "coverage_assessment",
                "exclusion_assessment",
                "obligation_assessment",
                "claim_calculation",
                "claim_recommendation",
            ],
            "current_phase": "claim_recommendation",
            "next_action": "completed",
        })

    async def _assert_database_mutation_rejected(self, statement) -> None:
        savepoint = await self.session.begin_nested()
        try:
            with self.assertRaises(DBAPIError):
                await self.session.execute(statement)
                await self.session.flush()
        finally:
            if savepoint.is_active:
                await savepoint.rollback()

    async def test_full_module4_lifecycle(self):
        initial, missing = self._initial_state()
        claim_id = initial.claim.id
        session_id = uuid.uuid4()

        # 4.1: persist and restore the authoritative mutable state.
        saved = await self.persistence.save_state(initial)
        self.assertEqual(saved.revision, 1)
        loaded = await self.persistence.load_latest_state(claim_id)
        self.assertEqual(
            loaded.state.model_dump(mode="json"),
            initial.model_dump(mode="json"),
        )
        self.assertEqual(loaded.state.claim.id, claim_id)

        # 4.2 + 4.3: append initial history and create immutable V1.
        initial_events = await self.history.record_initial_assessment(
            fnol_text=(
                "My bicycle was stolen outside Zurich station on 12 August 2026. "
                "I am claiming CHF 1800 for the stolen bicycle."
            ),
            state=initial,
            assessment_session_id=session_id,
        )
        fnol_event = next(item for item in initial_events if item.event_type == "fnol_submitted")
        v1 = await self.versions.create_version(
            initial,
            trigger_type="initial_assessment",
            trigger_reference_id=fnol_event.event_id,
        )
        v1_snapshot = json.loads(json.dumps(v1.phase_outputs))

        # Deterministic follow-up: retain raw text separately from normalized True.
        forbidden_orchestrator = AsyncMock()
        forbidden_orchestrator.assess.side_effect = AssertionError("Phase 1-7 must not run")
        agent = CaseLensAgent(forbidden_orchestrator, history_service=self.history)
        answer = MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
            missing_information_id=missing.id,
            value=True,
            original_text="The bicycle was locked to a rack.",
        )])
        events_before_answer = list(await self.history.list_events(claim_id))
        old_event_snapshots = {
            item.event_id: {
                "event_type": item.event_type,
                "field_path": item.field_path,
                "raw_answer": item.raw_answer,
                "new_value": item.new_value,
                "normalized_value": item.normalized_value,
            }
            for item in events_before_answer
        }
        answered = await agent.apply_missing_information_answers(initial, answer)
        locked_fact = next(
            item for item in answered.facts
            if item.fact_path == "incidents[0].metadata.bicycle_locked"
        )
        self.assertIs(locked_fact.value, True)
        self.assertEqual(answered.next_action, "rerun_phase")
        self.assertEqual(answered.current_phase, "exclusion_assessment")
        answer_events = list(await self.history.list_events(claim_id))
        answer_events_by_id = {item.event_id: item for item in answer_events}
        for event_id, snapshot in old_event_snapshots.items():
            item = answer_events_by_id[event_id]
            self.assertEqual({
                "event_type": item.event_type,
                "field_path": item.field_path,
                "raw_answer": item.raw_answer,
                "new_value": item.new_value,
                "normalized_value": item.normalized_value,
            }, snapshot)
        received = next(item for item in answer_events if item.event_type == "user_answer_received")
        applied = next(item for item in answer_events if item.event_type == "user_answer_applied")
        self.assertEqual(received.raw_answer, "The bicycle was locked to a rack.")
        self.assertIsNone(received.normalized_value)
        self.assertIs(applied.normalized_value, True)

        revision2 = await self.persistence.update_state(answered, expected_revision=1)
        self.assertEqual(revision2.revision, 2)

        # Controlled rerun test double is the only execution substitute.
        completed = self._completed_state(answered)
        rerunner = ControlledSelectiveRerunner(completed)
        registry = ToolRegistry()
        registry.register(ServiceTool(
            name="selective_rerun",
            description="Controlled Module 4 integration rerun.",
            execute=rerunner.rerun,
            phase_resolver=lambda state: state.current_phase,
        ))
        rerun_agent = CaseLensAgent(
            forbidden_orchestrator,
            tool_registry=registry,
            history_service=self.history,
            assessment_version_service=self.versions,
        )
        rerun_result = await rerun_agent.execute_next(answered)
        self.assertTrue(rerun_result.success)
        latest_state = rerun_result.output
        self.assertEqual(rerunner.calls, 1)
        self.assertEqual(latest_state.next_action, "completed")

        revision3 = await self.persistence.update_state(latest_state, expected_revision=2)
        self.assertEqual(revision3.revision, 3)

        versions = list(await self.versions.list_versions(claim_id))
        self.assertEqual([item.version_number for item in versions], [1, 2])
        v2 = versions[1]
        self.assertEqual(v2.parent_version_id, v1.assessment_version_id)
        self.assertEqual(v2.trigger_type, "selective_rerun")
        self.assertEqual(v2.rerun_from_phase, "exclusion_assessment")
        self.assertEqual(v1.phase_outputs, v1_snapshot)

        # Optimistic locking prevents an obsolete writer from replacing revision 3.
        with self.assertRaises(AgentStateRevisionConflictError):
            await self.persistence.update_state(answered, expected_revision=2)
        current = await self.persistence.load_latest_state(claim_id)
        self.assertEqual(current.revision, 3)
        self.assertEqual(
            current.state.model_dump(mode="json"),
            latest_state.model_dump(mode="json"),
        )

        # 4.4: resume is read-only and routes the restored completed state.
        resume = ClaimResumeService(self.persistence)
        resumed = await resume.resume(claim_id)
        self.assertEqual(
            resumed.state.model_dump(mode="json"),
            latest_state.model_dump(mode="json"),
        )
        self.assertEqual(resumed.next_action, "completed")
        forbidden_orchestrator.assess.assert_not_awaited()

        # 4.5: bounded phase context is deterministic and provenance preserving.
        context_service = ContextRetrievalService(
            self.persistence,
            self.event_repository,
            self.version_repository,
        )
        context = await context_service.retrieve_for_phase(
            claim_id, current_phase="claim_recommendation", max_items=5
        )
        context_again = await context_service.retrieve_for_phase(
            claim_id, current_phase="claim_recommendation", max_items=5
        )
        self.assertLessEqual(context.total_items, 5)
        self.assertEqual(
            [(item.source_type, item.source_id) for item in context.items],
            [(item.source_type, item.source_id) for item in context_again.items],
        )
        self.assertTrue(all(item.claim_id == claim_id for item in context.items))
        self.assertTrue(all(item.source_id is not None for item in context.items))
        self.assertIn("claim_event", {item.source_type for item in context.items})
        self.assertIn("assessment_version", {item.source_type for item in context.items})
        after_context = await self.persistence.load_latest_state(claim_id)
        self.assertEqual(
            after_context.state.model_dump(mode="json"),
            latest_state.model_dump(mode="json"),
        )
        self.assertEqual(after_context.revision, 3)

        # PostgreSQL triggers enforce append-only history and versions.
        event_count_before = len(await self.history.list_events(claim_id))
        await self._assert_database_mutation_rejected(
            update(ClaimEventRecord)
            .where(ClaimEventRecord.event_id == fnol_event.event_id)
            .values(raw_answer="mutated")
        )
        await self._assert_database_mutation_rejected(
            delete(ClaimEventRecord).where(ClaimEventRecord.event_id == fnol_event.event_id)
        )
        await self._assert_database_mutation_rejected(
            update(AssessmentVersionRecord)
            .where(AssessmentVersionRecord.assessment_version_id == v1.assessment_version_id)
            .values(status="failed")
        )
        await self._assert_database_mutation_rejected(
            delete(AssessmentVersionRecord).where(
                AssessmentVersionRecord.assessment_version_id == v1.assessment_version_id
            )
        )
        self.assertEqual(len(await self.history.list_events(claim_id)), event_count_before)
        self.assertEqual((await self.versions.get_version(v1.assessment_version_id)).phase_outputs, v1_snapshot)

        summary = {
            "claim_id": str(claim_id),
            "current_revision": current.revision,
            "event_count": event_count_before,
            "assessment_versions": [item.version_number for item in versions],
            "latest_version_number": v2.version_number,
            "resumed_next_action": resumed.next_action,
            "retrieved_context_item_count": context.total_items,
        }
        print("MODULE4_LIFECYCLE_SUMMARY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
