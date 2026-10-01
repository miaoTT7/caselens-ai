"""Persistence boundary for the current CaseLens agent-state snapshot."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from api.agent import CaseLensAgentState
from api.models import ClaimAgentStateRecord
from api.repositories import ClaimAgentStateRepository


CURRENT_STATE_SCHEMA_VERSION = 1


class UnsupportedAgentStateSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class PersistedAgentState:
    state: CaseLensAgentState
    revision: int
    state_schema_version: int
    created_at: datetime | None
    updated_at: datetime | None


class ClaimAgentStatePersistenceService:
    """Serialize, validate, save, and restore the one current state per claim."""

    def __init__(self, repository: ClaimAgentStateRepository):
        self.repository = repository

    async def save_state(self, state: CaseLensAgentState) -> PersistedAgentState:
        claim_id, values = self._serialize(state)
        record = await self.repository.create(
            claim_id=claim_id,
            **values,
            state_schema_version=CURRENT_STATE_SCHEMA_VERSION,
            revision=1,
        )
        return self._restore(record)

    async def load_latest_state(self, claim_id: uuid.UUID) -> PersistedAgentState | None:
        record = await self.repository.get(claim_id)
        return None if record is None else self._restore(record)

    async def update_state(
        self,
        state: CaseLensAgentState,
        *,
        expected_revision: int,
    ) -> PersistedAgentState:
        if expected_revision < 1:
            raise ValueError("expected_revision must be at least 1")
        claim_id, values = self._serialize(state)
        record = await self.repository.update(
            claim_id=claim_id,
            expected_revision=expected_revision,
            values={
                **values,
                "state_schema_version": CURRENT_STATE_SCHEMA_VERSION,
            },
        )
        return self._restore(record)

    @staticmethod
    def _serialize(state: CaseLensAgentState) -> tuple[uuid.UUID, dict[str, Any]]:
        if state.claim is None:
            raise ValueError("Cannot persist agent state without a claim")

        # JSON mode is intentional: UUID, Decimal, date, and datetime values
        # must cross the JSONB boundary without driver-specific conversion.
        payload = state.model_dump(mode="json")
        claim_id = state.claim.id
        serialized_claim_id = payload["claim"].get("id")
        if serialized_claim_id != str(claim_id):
            raise ValueError("Agent state claim.id does not match the persistence claim_id")

        return claim_id, {
            "knowledge_base_id": state.knowledge_base_id,
            "claim": payload["claim"],
            "facts": payload["facts"],
            "missing_information": payload["missing_information"],
            "applicable_policy": payload["applicable_policy"],
            "coverage_assessments": payload["coverage_assessments"],
            "exclusion_assessments": payload["exclusion_assessments"],
            "obligation_assessments": payload["obligation_assessments"],
            "calculation_results": payload["calculation_results"],
            "recommendation": payload["recommendation"],
            "current_phase": payload["current_phase"],
            "next_action": payload["next_action"],
            "completed_phases": payload["completed_phases"],
            "provider_errors": payload["provider_errors"],
            "rerun_failure_count": payload["rerun_failure_count"],
            "retrieval_limit": payload["retrieval_limit"],
            "claimant_reference_required": payload["claimant_reference_required"],
            "policy_reference_required": payload["policy_reference_required"],
        }

    @staticmethod
    def _restore(record: ClaimAgentStateRecord) -> PersistedAgentState:
        if record.state_schema_version != CURRENT_STATE_SCHEMA_VERSION:
            raise UnsupportedAgentStateSchemaError(
                f"Unsupported agent state schema version: {record.state_schema_version}"
            )

        payload = {
            "knowledge_base_id": record.knowledge_base_id,
            "retrieval_limit": record.retrieval_limit,
            "claimant_reference_required": record.claimant_reference_required,
            "policy_reference_required": record.policy_reference_required,
            "claim": record.claim,
            "facts": record.facts,
            "missing_information": record.missing_information,
            "applicable_policy": record.applicable_policy,
            "coverage_assessments": record.coverage_assessments,
            "exclusion_assessments": record.exclusion_assessments,
            "obligation_assessments": record.obligation_assessments,
            "calculation_results": record.calculation_results,
            "recommendation": record.recommendation,
            "current_phase": record.current_phase,
            "next_action": record.next_action,
            "completed_phases": record.completed_phases,
            "provider_errors": record.provider_errors,
            "rerun_failure_count": record.rerun_failure_count,
        }
        state = CaseLensAgentState.model_validate(payload)
        if state.claim is None or state.claim.id != record.claim_id:
            raise ValueError("Persisted state claim.id does not match claim_id")

        return PersistedAgentState(
            state=state,
            revision=record.revision,
            state_schema_version=record.state_schema_version,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
