import unittest
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from api.agent import CaseLensAgentState
from api.agent_state_persistence import (
    CURRENT_STATE_SCHEMA_VERSION,
    ClaimAgentStatePersistenceService,
    UnsupportedAgentStateSchemaError,
)
from api.claim_schemas import Claim, ClaimFact, ClaimRecommendation
from api.models import ClaimAgentStateRecord
from api.repositories import AgentStateRevisionConflictError


class FakeAgentStateRepository:
    def __init__(self):
        self.records: dict[uuid.UUID, ClaimAgentStateRecord] = {}

    async def create(self, **values):
        claim_id = values["claim_id"]
        if claim_id in self.records:
            raise RuntimeError("duplicate state")
        record = ClaimAgentStateRecord(**values)
        now = datetime.now(UTC)
        record.created_at = now
        record.updated_at = now
        self.records[claim_id] = record
        return record

    async def get(self, claim_id):
        return self.records.get(claim_id)

    async def update(self, *, claim_id, expected_revision, values):
        record = self.records.get(claim_id)
        if record is None:
            raise LookupError(f"No persisted agent state for claim {claim_id}")
        if record.revision != expected_revision:
            raise AgentStateRevisionConflictError("stale revision")
        for name, value in values.items():
            setattr(record, name, value)
        record.revision += 1
        record.updated_at = datetime.now(UTC)
        return record


def build_state() -> CaseLensAgentState:
    claim_id = uuid.uuid4()
    fact = ClaimFact(
        id=uuid.uuid4(),
        fact_path="claim.claimed_amount",
        value="1800.00",
        status="confirmed",
    )
    recommendation = ClaimRecommendation(
        id=uuid.uuid4(),
        status="needs_information",
        recommended_payable_amount=Decimal("1600.00"),
        currency="CHF",
        reasons=["Additional information is required."],
        human_review_required=False,
    )
    return CaseLensAgentState(
        knowledge_base_id=uuid.uuid4(),
        retrieval_limit=12,
        claimant_reference_required=True,
        claim=Claim(id=claim_id, description="Bicycle theft"),
        facts=[fact],
        recommendation=recommendation,
        current_phase="claim_recommendation",
        next_action="ask_for_information",
        completed_phases=["claim_facts_extraction", "fact_validation"],
        provider_errors=["sanitized provider error"],
        rerun_failure_count=2,
    )


class AgentStatePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.repository = FakeAgentStateRepository()
        self.service = ClaimAgentStatePersistenceService(self.repository)

    async def test_save_and_load_restore_the_exact_validated_state(self):
        state = build_state()

        saved = await self.service.save_state(state)
        loaded = await self.service.load_latest_state(state.claim.id)

        self.assertEqual(saved.revision, 1)
        self.assertEqual(saved.state_schema_version, CURRENT_STATE_SCHEMA_VERSION)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.state, state)
        self.assertEqual(
            loaded.state.recommendation.recommended_payable_amount,
            Decimal("1600.00"),
        )

    async def test_update_requires_current_revision_and_increments_it(self):
        state = build_state()
        await self.service.save_state(state)
        changed = state.model_copy(update={"next_action": "human_review"})

        updated = await self.service.update_state(changed, expected_revision=1)

        self.assertEqual(updated.revision, 2)
        self.assertEqual(updated.state.next_action, "human_review")
        with self.assertRaises(AgentStateRevisionConflictError):
            await self.service.update_state(state, expected_revision=1)

    async def test_save_rejects_state_without_claim(self):
        with self.assertRaisesRegex(ValueError, "without a claim"):
            await self.service.save_state(CaseLensAgentState())

    async def test_load_rejects_claim_id_mismatch(self):
        state = build_state()
        await self.service.save_state(state)
        record = self.repository.records[state.claim.id]
        record.claim = {**record.claim, "id": str(uuid.uuid4())}

        with self.assertRaisesRegex(ValueError, "does not match claim_id"):
            await self.service.load_latest_state(state.claim.id)

    async def test_load_rejects_unsupported_schema_version(self):
        state = build_state()
        await self.service.save_state(state)
        self.repository.records[state.claim.id].state_schema_version = 99

        with self.assertRaises(UnsupportedAgentStateSchemaError):
            await self.service.load_latest_state(state.claim.id)


if __name__ == "__main__":
    unittest.main()
