import unittest
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.agent import CaseLensAgentState
from api.claim_schemas import Claim, MissingInformation
from api.context_retrieval import ContextRetrievalService


def event(
    claim_id,
    *,
    event_type="claim_fact_updated",
    phase="claim_calculation",
    field_path="claim.claimed_amount",
    timestamp=None,
    raw_answer=None,
    normalized_value=None,
    missing_information_id=None,
):
    return SimpleNamespace(
        event_id=uuid.uuid4(),
        claim_id=claim_id,
        assessment_session_id=uuid.uuid4(),
        event_type=event_type,
        actor_type="system",
        field_path=field_path,
        old_value=None,
        new_value=normalized_value,
        raw_answer=raw_answer,
        normalized_value=normalized_value,
        missing_information_id=missing_information_id,
        evidence_refs=[],
        related_phase=phase,
        created_at=timestamp or datetime.now(UTC),
    )


def version(claim_id, number, timestamp, phase="claim_calculation"):
    return SimpleNamespace(
        assessment_version_id=uuid.uuid4(),
        claim_id=claim_id,
        parent_version_id=None,
        version_number=number,
        trigger_type="selective_rerun" if number > 1 else "initial_assessment",
        trigger_reference_id=uuid.uuid4(),
        rerun_from_phase=phase,
        completed_phases=[phase],
        phase_outputs={
            "coverage_assessments": [{"status": "covered"}],
            "calculation_results": [{"status": "complete", "payable_amount": "1600"}],
            "missing_information": [],
            "provider_errors": [],
        },
        recommendation={"status": "recommend_approve"},
        status="completed",
        created_at=timestamp,
    )


class ContextRetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.claim_id = uuid.uuid4()
        self.persistence = AsyncMock()
        self.events = AsyncMock()
        self.versions = AsyncMock()
        self.events.list_context_events.return_value = []
        self.versions.get_latest_version.return_value = None
        self.versions.list_context_versions.return_value = []
        self.service = ContextRetrievalService(
            self.persistence, self.events, self.versions
        )

    def set_state(self, phase, missing=None):
        state = CaseLensAgentState(
            claim=Claim(id=self.claim_id),
            current_phase=phase,
            missing_information=missing or [],
        )
        self.persistence.load_latest_state.return_value = SimpleNamespace(state=state)
        return state

    async def test_phase_scoped_retrieval_uses_exact_current_missing_ids(self):
        missing = MissingInformation(
            id=uuid.uuid4(),
            field_path="claim.eligible_amount",
            reason="Eligible amount required.",
            required_for="claim_calculation",
            blocking=True,
        )
        self.set_state("claim_calculation", [missing])
        relevant = event(
            self.claim_id,
            field_path="claim.eligible_amount",
            missing_information_id=missing.id,
            normalized_value="1800",
        )
        self.events.list_context_events.return_value = [relevant]

        bundle = await self.service.retrieve_for_phase(self.claim_id, max_items=5)

        kwargs = self.events.list_context_events.await_args.kwargs
        self.assertEqual(kwargs["missing_information_ids"], [missing.id])
        self.assertIn("claim.claimed_amount", kwargs["field_path_patterns"])
        self.assertEqual(kwargs["related_phases"], ("claim_calculation",))
        self.assertEqual(kwargs["limit"], 6)
        self.assertEqual(bundle.items[0].source_id, relevant.event_id)

    async def test_relevance_then_newest_order_is_deterministic(self):
        self.set_state("claim_calculation")
        now = datetime.now(UTC)
        fact_old = event(
            self.claim_id,
            timestamp=now - timedelta(hours=2),
            normalized_value="1800",
        )
        fact_new = event(
            self.claim_id,
            timestamp=now - timedelta(hours=1),
            field_path="claim.currency",
            normalized_value="CHF",
        )
        provider_newest = event(
            self.claim_id,
            event_type="provider_error",
            timestamp=now,
            field_path=None,
        )
        self.events.list_context_events.return_value = [provider_newest, fact_old, fact_new]

        bundle = await self.service.retrieve_for_phase(self.claim_id, max_items=5)

        self.assertEqual(
            [item.source_id for item in bundle.items],
            [fact_new.event_id, fact_old.event_id, provider_newest.event_id],
        )

    async def test_max_items_truncates_without_unbounded_queries(self):
        self.set_state("fact_validation")
        now = datetime.now(UTC)
        rows = [
            event(self.claim_id, phase="fact_validation", timestamp=now - timedelta(minutes=i))
            for i in range(4)
        ]
        self.events.list_context_events.return_value = rows

        bundle = await self.service.retrieve_for_phase(self.claim_id, max_items=2)

        self.assertEqual(bundle.total_items, 2)
        self.assertTrue(bundle.truncated)
        self.assertEqual(self.events.list_context_events.await_args.kwargs["limit"], 3)
        self.versions.list_context_versions.assert_not_awaited()

    async def test_version_and_event_provenance_are_preserved(self):
        self.set_state("claim_calculation")
        now = datetime.now(UTC)
        event_row = event(self.claim_id, timestamp=now)
        version_row = version(self.claim_id, 2, now - timedelta(minutes=1))
        self.events.list_context_events.return_value = [event_row]
        self.versions.get_latest_version.return_value = version_row
        self.versions.list_context_versions.return_value = [version_row]

        bundle = await self.service.retrieve_for_phase(self.claim_id, max_items=5)

        by_type = {item.source_type: item for item in bundle.items}
        self.assertEqual(by_type["claim_event"].source_id, event_row.event_id)
        self.assertEqual(
            by_type["assessment_version"].source_id,
            version_row.assessment_version_id,
        )
        self.assertEqual(
            by_type["assessment_version"].metadata["trigger_reference_id"],
            str(version_row.trigger_reference_id),
        )

    async def test_empty_history_returns_empty_bundle(self):
        self.set_state("fact_validation")

        bundle = await self.service.retrieve_for_phase(self.claim_id)

        self.assertEqual(bundle.items, [])
        self.assertEqual(bundle.total_items, 0)
        self.assertFalse(bundle.truncated)

    async def test_timeline_is_chronological_and_bounded(self):
        self.set_state("obligation_assessment")
        now = datetime.now(UTC)
        anchor = event(self.claim_id, timestamp=now, phase="obligation_assessment")
        older_near = event(
            self.claim_id, timestamp=now - timedelta(minutes=1), phase="obligation_assessment"
        )
        older_far = event(
            self.claim_id, timestamp=now - timedelta(minutes=2), phase="obligation_assessment"
        )
        newer = event(
            self.claim_id, timestamp=now + timedelta(minutes=1), phase="obligation_assessment"
        )
        extra = event(
            self.claim_id, timestamp=now + timedelta(minutes=2), phase="obligation_assessment"
        )
        self.events.get_event.return_value = anchor
        self.events.list_events_before.return_value = [older_near, older_far]
        self.events.list_events_after.return_value = [newer, extra]

        bundle = await self.service.retrieve_timeline(
            self.claim_id, anchor_event_id=anchor.event_id, before=1, after=1
        )

        self.assertEqual(
            [item.source_id for item in bundle.items],
            [older_near.event_id, anchor.event_id, newer.event_id],
        )
        self.assertTrue(bundle.truncated)
        self.assertEqual(self.events.list_events_before.await_args.kwargs["limit"], 2)
        self.assertEqual(self.events.list_events_after.await_args.kwargs["limit"], 2)

    async def test_text_search_is_bounded_and_preserves_sources(self):
        self.set_state("claim_recommendation")
        now = datetime.now(UTC)
        answer = event(
            self.claim_id,
            event_type="user_answer_received",
            phase="claim_recommendation",
            raw_answer="The bicycle was locked.",
            timestamp=now,
        )
        old_version = version(
            self.claim_id, 1, now - timedelta(minutes=1), "claim_recommendation"
        )
        self.events.search_text_events.return_value = [answer]
        self.versions.search_text_versions.return_value = [old_version]

        bundle = await self.service.search_text_history(
            self.claim_id, query="locked", max_items=2
        )

        self.assertEqual(bundle.total_items, 2)
        self.assertEqual(bundle.items[0].source_id, answer.event_id)
        self.assertEqual(self.events.search_text_events.await_args.kwargs["limit"], 3)
        self.assertEqual(self.versions.search_text_versions.await_args.kwargs["limit"], 3)

    async def test_context_retrieval_has_no_business_or_provider_calls(self):
        self.set_state("coverage_assessment")
        provider = AsyncMock()
        llm = AsyncMock()
        business_service = AsyncMock()
        tool_registry = AsyncMock()

        await self.service.retrieve_for_phase(self.claim_id)

        provider.assert_not_called()
        llm.assert_not_called()
        business_service.assert_not_called()
        tool_registry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
