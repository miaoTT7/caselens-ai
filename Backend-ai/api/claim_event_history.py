"""Append-only claim fact and interaction history."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from api.agent import (
    CaseLensAgentState,
    MissingInformationAnswerRequest,
    PhaseName,
)
from api.claim_schemas import EvidenceRef
from api.human_handoff import HumanReviewHandoff
from api.models import ClaimEventRecord
from api.repositories import ClaimEventRepository


ClaimEventType = Literal[
    "fnol_submitted",
    "claim_fact_extracted",
    "claim_fact_updated",
    "missing_information_created",
    "user_answer_received",
    "user_answer_applied",
    "provider_error",
    "phase_completed",
    "selective_rerun_started",
    "selective_rerun_completed",
    "human_handoff_created",
]
ActorType = Literal["user", "agent", "system", "provider", "human_reviewer"]


class ClaimEventCreate(BaseModel):
    claim_id: uuid.UUID
    assessment_session_id: uuid.UUID | None = None
    event_type: ClaimEventType
    actor_type: ActorType
    field_path: str | None = None
    old_value: Any | None = None
    new_value: Any | None = None
    raw_answer: str | None = None
    normalized_value: Any | None = None
    missing_information_id: uuid.UUID | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    related_phase: PhaseName | None = None


class ClaimEventPersistenceError(RuntimeError):
    pass


class ClaimEventHistoryService:
    """Record existing business outcomes without deriving new conclusions."""

    _json_adapter = TypeAdapter(Any)

    def __init__(self, repository: ClaimEventRepository):
        self.repository = repository

    async def append_event(self, event: ClaimEventCreate) -> ClaimEventRecord:
        return (await self._append_events([event]))[0]

    async def list_events(self, claim_id: uuid.UUID) -> Sequence[ClaimEventRecord]:
        return await self.repository.list_events(claim_id)

    async def list_fact_history(
        self, claim_id: uuid.UUID, field_path: str
    ) -> Sequence[ClaimEventRecord]:
        return await self.repository.list_fact_history(claim_id, field_path)

    async def record_initial_assessment(
        self,
        *,
        fnol_text: str,
        state: CaseLensAgentState,
        assessment_session_id: uuid.UUID | None,
    ) -> list[ClaimEventRecord]:
        claim_id = self._claim_id(state)
        common = {"claim_id": claim_id, "assessment_session_id": assessment_session_id}
        events = [ClaimEventCreate(
            **common,
            event_type="fnol_submitted",
            actor_type="user",
            raw_answer=fnol_text,
            related_phase="claim_facts_extraction",
        )]
        events.extend(ClaimEventCreate(
            **common,
            event_type="claim_fact_extracted",
            actor_type="agent",
            field_path=fact.fact_path,
            new_value=self._json(fact.value),
            normalized_value=self._json(fact.value),
            evidence_refs=fact.claim_evidence,
            related_phase="claim_facts_extraction",
        ) for fact in state.facts)
        events.extend(ClaimEventCreate(
            **common,
            event_type="missing_information_created",
            actor_type="agent",
            field_path=item.field_path,
            new_value=item.model_dump(mode="json"),
            missing_information_id=item.id,
            related_phase=self._phase_for_requirement(item.required_for),
        ) for item in state.missing_information)
        events.extend(ClaimEventCreate(
            **common,
            event_type="phase_completed",
            actor_type="system",
            related_phase=phase,
        ) for phase in state.completed_phases)
        events.extend(ClaimEventCreate(
            **common,
            event_type="provider_error",
            actor_type="provider",
            new_value=error,
            related_phase=state.current_phase,
        ) for error in state.provider_errors)
        return await self._append_events(events)

    async def record_answers(
        self,
        *,
        before: CaseLensAgentState,
        after: CaseLensAgentState,
        request: MissingInformationAnswerRequest,
        assessment_session_id: uuid.UUID | None,
    ) -> None:
        claim_id = self._claim_id(after)
        missing_by_id = {item.id: item for item in before.missing_information}
        old_facts = {item.fact_path: item for item in before.facts}
        new_facts = {item.fact_path: item for item in after.facts}
        events: list[ClaimEventCreate] = []
        for answer in request.answers:
            missing = missing_by_id[answer.missing_information_id]
            fact = new_facts[missing.field_path]
            raw = answer.original_text if answer.original_text is not None else str(answer.value)
            common = {
                "claim_id": claim_id,
                "assessment_session_id": assessment_session_id,
                "field_path": missing.field_path,
                "missing_information_id": missing.id,
                "related_phase": self._phase_for_requirement(missing.required_for),
            }
            events.append(ClaimEventCreate(
                **common,
                event_type="user_answer_received",
                actor_type="user",
                raw_answer=raw,
            ))
            events.append(ClaimEventCreate(
                **common,
                event_type="user_answer_applied",
                actor_type="system",
                raw_answer=raw,
                normalized_value=self._json(fact.value),
                evidence_refs=fact.claim_evidence,
            ))
            old = old_facts.get(fact.fact_path)
            if old is None or old.value != fact.value or old.status != fact.status:
                events.append(ClaimEventCreate(
                    **common,
                    event_type="claim_fact_updated",
                    actor_type="system",
                    old_value=self._json(old.value) if old is not None else None,
                    new_value=self._json(fact.value),
                    normalized_value=self._json(fact.value),
                    evidence_refs=fact.claim_evidence,
                ))
        await self._append_events(events)

    async def record_rerun_started(
        self, state: CaseLensAgentState, assessment_session_id: uuid.UUID | None
    ) -> ClaimEventRecord:
        return await self.append_event(ClaimEventCreate(
            claim_id=self._claim_id(state),
            assessment_session_id=assessment_session_id,
            event_type="selective_rerun_started",
            actor_type="system",
            related_phase=state.current_phase,
        ))

    async def record_rerun_completed(
        self,
        *,
        before: CaseLensAgentState,
        after: CaseLensAgentState | None,
        success: bool,
        error: str | None,
        assessment_session_id: uuid.UUID | None,
    ) -> list[ClaimEventRecord]:
        effective = after or before
        events = [ClaimEventCreate(
            claim_id=self._claim_id(before),
            assessment_session_id=assessment_session_id,
            event_type="selective_rerun_completed",
            actor_type="system",
            related_phase=before.current_phase,
            new_value={
                "success": success,
                "error": error,
                "current_phase": effective.current_phase,
                "next_action": effective.next_action,
                "rerun_failure_count": effective.rerun_failure_count,
            },
        )]
        old_missing_ids = {item.id for item in before.missing_information}
        events.extend(ClaimEventCreate(
            claim_id=self._claim_id(before),
            assessment_session_id=assessment_session_id,
            event_type="missing_information_created",
            actor_type="agent",
            field_path=item.field_path,
            new_value=item.model_dump(mode="json"),
            missing_information_id=item.id,
            related_phase=self._phase_for_requirement(item.required_for),
        ) for item in effective.missing_information if item.id not in old_missing_ids)
        for provider_error in effective.provider_errors[len(before.provider_errors):]:
            events.append(ClaimEventCreate(
                claim_id=self._claim_id(before),
                assessment_session_id=assessment_session_id,
                event_type="provider_error",
                actor_type="provider",
                new_value=provider_error,
                related_phase=effective.current_phase,
            ))
        return await self._append_events(events)

    async def record_handoff(
        self,
        *,
        state: CaseLensAgentState,
        handoff: HumanReviewHandoff,
        assessment_session_id: uuid.UUID | None,
    ) -> None:
        await self.append_event(ClaimEventCreate(
            claim_id=self._claim_id(state),
            assessment_session_id=assessment_session_id,
            event_type="human_handoff_created",
            actor_type="system",
            related_phase=state.current_phase,
            new_value={
                "reason": handoff.reason,
                "suggested_review_focus": handoff.suggested_review_focus,
            },
            evidence_refs=handoff.relevant_evidence,
        ))

    async def _append_events(
        self, events: Sequence[ClaimEventCreate]
    ) -> list[ClaimEventRecord]:
        if not events:
            return []
        values = []
        for event in events:
            value = event.model_dump(mode="json")
            value.update({
                "claim_id": event.claim_id,
                "assessment_session_id": event.assessment_session_id,
                "missing_information_id": event.missing_information_id,
            })
            values.append(value)
        try:
            return await self.repository.append_events(values)
        except Exception as error:
            claim_id = events[0].claim_id
            event_types = ", ".join(dict.fromkeys(item.event_type for item in events))
            raise ClaimEventPersistenceError(
                f"Failed to persist claim history for {claim_id} ({event_types})"
            ) from error

    @classmethod
    def _json(cls, value: Any) -> Any:
        return cls._json_adapter.dump_python(value, mode="json")

    @staticmethod
    def _claim_id(state: CaseLensAgentState) -> uuid.UUID:
        if state.claim is None:
            raise ValueError("Claim history requires an agent state with a claim")
        return state.claim.id

    @staticmethod
    def _phase_for_requirement(required_for: str) -> PhaseName | None:
        mapping: dict[str, PhaseName] = {
            "fact_validation": "fact_validation",
            "policy_applicability": "coverage_assessment",
            "coverage_assessment": "coverage_assessment",
            "exclusion_assessment": "exclusion_assessment",
            "obligation_assessment": "obligation_assessment",
            "claim_calculation": "claim_calculation",
        }
        return mapping.get(required_for)
