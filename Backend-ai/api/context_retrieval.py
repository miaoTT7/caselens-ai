"""Bounded, read-only historical context retrieval for claim workflows."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from api.agent import CaseLensAgentState, PhaseName
from api.agent_state_persistence import ClaimAgentStatePersistenceService
from api.models import AssessmentVersionRecord, ClaimEventRecord
from api.repositories import AssessmentVersionRepository, ClaimEventRepository


ContextSourceType = Literal["claim_event", "assessment_version"]


class ContextItem(BaseModel):
    source_type: ContextSourceType
    source_id: uuid.UUID
    claim_id: uuid.UUID
    related_phase: PhaseName | None = None
    field_path: str | None = None
    timestamp: datetime
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBundle(BaseModel):
    current_phase: PhaseName
    items: list[ContextItem] = Field(default_factory=list)
    total_items: int = Field(ge=0)
    truncated: bool


class ContextStateNotFoundError(LookupError):
    pass


class ContextRetrievalService:
    _EVENT_TYPES: dict[PhaseName, tuple[str, ...]] = {
        "claim_facts_extraction": (
            "fnol_submitted", "claim_fact_extracted", "claim_fact_updated"
        ),
        "fact_validation": (
            "claim_fact_extracted", "claim_fact_updated",
            "missing_information_created", "user_answer_received", "user_answer_applied",
        ),
        "coverage_assessment": (
            "claim_fact_extracted", "claim_fact_updated", "user_answer_applied",
            "missing_information_created", "phase_completed",
            "selective_rerun_completed", "provider_error",
        ),
        "exclusion_assessment": (
            "claim_fact_updated", "user_answer_received", "user_answer_applied",
            "missing_information_created", "provider_error",
            "selective_rerun_started", "selective_rerun_completed", "phase_completed",
        ),
        "obligation_assessment": (
            "claim_fact_updated", "user_answer_received", "user_answer_applied",
            "missing_information_created", "provider_error",
            "selective_rerun_completed", "phase_completed",
        ),
        "claim_calculation": (
            "claim_fact_extracted", "claim_fact_updated", "user_answer_received",
            "user_answer_applied", "missing_information_created", "provider_error",
            "selective_rerun_completed", "phase_completed",
        ),
        "claim_recommendation": (
            "provider_error", "human_handoff_created", "selective_rerun_started",
            "selective_rerun_completed", "phase_completed", "missing_information_created",
        ),
    }
    _FIELD_PATTERNS: dict[PhaseName, tuple[str, ...]] = {
        "claim_facts_extraction": ("claim.%", "incidents[%", "exposures[%"),
        "fact_validation": (
            "claim.incidents", "%incident_type", "%event_date", "%location",
            "claim.claimed_amount", "claim.currency", "claim.claimant_reference",
            "claim.policy%",
        ),
        "coverage_assessment": (
            "%incident_type", "%event_date", "%location", "%cause", "%description",
            "%metadata.%", "claim.policy%", "claim.claimed_amount", "claim.currency",
            "exposures%", "coverage.%",
        ),
        "exclusion_assessment": (
            "%cause", "%location", "%description", "%metadata.%", "exclusions.%",
            "%negligence%", "%security%", "%locked%", "%intent%",
        ),
        "obligation_assessment": (
            "obligations.%", "claim.reported_date", "claim.notification%",
            "claim.police_report%", "claim.documents%", "claim.cooperation%",
            "%metadata.%", "%notify%", "%report%", "%police%", "%mitigation%",
            "%document%", "%cooperation%", "%locked%",
        ),
        "claim_calculation": (
            "claim.claimed_amount", "claim.eligible_amount", "claim.currency",
            "%deductible%", "%excess%", "%limit%", "%sublimit%",
            "%other_insurance%", "calculation%", "exposures%claimed_amount",
            "exposures%currency",
        ),
        "claim_recommendation": (),
    }
    _VERSION_KEYS: dict[PhaseName, tuple[str, ...]] = {
        "claim_facts_extraction": (),
        "fact_validation": (),
        "coverage_assessment": (
            "applicable_policy", "coverage_assessments", "missing_information"
        ),
        "exclusion_assessment": (
            "coverage_assessments", "exclusion_assessments", "missing_information"
        ),
        "obligation_assessment": (
            "coverage_assessments", "exclusion_assessments",
            "obligation_assessments", "missing_information",
        ),
        "claim_calculation": (
            "coverage_assessments", "calculation_results", "missing_information"
        ),
        "claim_recommendation": (
            "coverage_assessments", "exclusion_assessments",
            "obligation_assessments", "calculation_results",
            "missing_information", "provider_errors",
        ),
    }
    _TEXT_EVENT_TYPES = (
        "fnol_submitted", "user_answer_received", "user_answer_applied",
        "human_handoff_created", "provider_error",
    )

    def __init__(
        self,
        state_persistence: ClaimAgentStatePersistenceService,
        event_repository: ClaimEventRepository,
        version_repository: AssessmentVersionRepository,
    ):
        self.state_persistence = state_persistence
        self.event_repository = event_repository
        self.version_repository = version_repository

    async def retrieve_for_phase(
        self,
        claim_id: uuid.UUID,
        *,
        current_phase: PhaseName | None = None,
        max_items: int = 20,
    ) -> ContextBundle:
        self._validate_limit(max_items)
        state = await self._load_state(claim_id)
        phase = current_phase or state.current_phase
        if phase is None:
            raise ValueError("A current phase is required for context retrieval")

        relevant_missing_ids = [
            item.id for item in state.missing_information
            if item.blocking or self._requirement_phase(item.required_for) == phase
        ]
        events = await self.event_repository.list_context_events(
            claim_id,
            event_types=self._EVENT_TYPES[phase],
            related_phases=(phase,),
            field_path_patterns=self._FIELD_PATTERNS[phase],
            missing_information_ids=relevant_missing_ids,
            limit=max_items + 1,
        )

        candidates: list[tuple[int, ContextItem]] = []
        missing_id_set = set(relevant_missing_ids)
        for event in events:
            priority = 0 if (
                event.missing_information_id in missing_id_set
                or event.event_type in {"claim_fact_extracted", "claim_fact_updated", "user_answer_applied"}
            ) else 1
            candidates.append((priority, self._event_item(event)))

        version_keys = self._VERSION_KEYS[phase]
        if version_keys:
            latest = await self.version_repository.get_latest_version(claim_id)
            versions = await self.version_repository.list_context_versions(
                claim_id, related_phases=(phase,), limit=max_items + 1
            )
            if latest is not None and all(
                item.assessment_version_id != latest.assessment_version_id
                for item in versions
            ):
                versions = [latest, *versions]
            for version in versions:
                priority = 0 if (
                    latest is not None
                    and version.assessment_version_id == latest.assessment_version_id
                ) else 2
                candidates.append((priority, self._version_item(version, phase)))

        return self._ranked_bundle(phase, candidates, max_items)

    async def retrieve_timeline(
        self,
        claim_id: uuid.UUID,
        *,
        anchor_event_id: uuid.UUID,
        before: int = 3,
        after: int = 3,
    ) -> ContextBundle:
        self._validate_window(before, after)
        state = await self._load_state(claim_id)
        if state.current_phase is None:
            raise ValueError("A current phase is required for timeline retrieval")
        anchor = await self.event_repository.get_event(claim_id, anchor_event_id)
        if anchor is None:
            raise LookupError(f"Claim event {anchor_event_id} was not found")
        older = await self.event_repository.list_events_before(
            claim_id,
            anchor_time=anchor.created_at,
            anchor_id=anchor.event_id,
            limit=before + 1,
        )
        newer = await self.event_repository.list_events_after(
            claim_id,
            anchor_time=anchor.created_at,
            anchor_id=anchor.event_id,
            limit=after + 1,
        )
        truncated = len(older) > before or len(newer) > after
        selected = [*list(reversed(older[:before])), anchor, *newer[:after]]
        items = [self._event_item(event) for event in selected]
        return ContextBundle(
            current_phase=state.current_phase,
            items=items,
            total_items=len(items),
            truncated=truncated,
        )

    async def search_text_history(
        self,
        claim_id: uuid.UUID,
        *,
        query: str,
        current_phase: PhaseName | None = None,
        max_items: int = 10,
    ) -> ContextBundle:
        self._validate_limit(max_items)
        if not query.strip():
            raise ValueError("History text query must not be empty")
        state = await self._load_state(claim_id)
        phase = current_phase or state.current_phase
        if phase is None:
            raise ValueError("A current phase is required for context retrieval")
        events = await self.event_repository.search_text_events(
            claim_id,
            query=query.strip(),
            event_types=self._TEXT_EVENT_TYPES,
            limit=max_items + 1,
        )
        versions = await self.version_repository.search_text_versions(
            claim_id, query=query.strip(), limit=max_items + 1
        )
        candidates = [
            (3, self._event_item(event)) for event in events
        ] + [
            (3, self._version_item(version, phase)) for version in versions
        ]
        return self._ranked_bundle(phase, candidates, max_items)

    async def _load_state(self, claim_id: uuid.UUID) -> CaseLensAgentState:
        persisted = await self.state_persistence.load_latest_state(claim_id)
        if persisted is None:
            raise ContextStateNotFoundError(
                f"No persisted agent state for claim {claim_id}"
            )
        state = persisted.state
        if state.claim is None or state.claim.id != claim_id:
            raise ValueError("Persisted state claim.id does not match claim_id")
        return state

    def _event_item(self, event: ClaimEventRecord) -> ContextItem:
        return ContextItem(
            source_type="claim_event",
            source_id=event.event_id,
            claim_id=event.claim_id,
            related_phase=event.related_phase,
            field_path=event.field_path,
            timestamp=event.created_at,
            content={
                "event_type": event.event_type,
                "raw_answer": event.raw_answer,
                "normalized_value": event.normalized_value,
                "old_value": event.old_value,
                "new_value": event.new_value,
                "evidence_refs": event.evidence_refs,
            },
            metadata={
                "actor_type": event.actor_type,
                "assessment_session_id": (
                    str(event.assessment_session_id)
                    if event.assessment_session_id is not None else None
                ),
                "missing_information_id": (
                    str(event.missing_information_id)
                    if event.missing_information_id is not None else None
                ),
            },
        )

    def _version_item(
        self, version: AssessmentVersionRecord, phase: PhaseName
    ) -> ContextItem:
        keys = self._VERSION_KEYS[phase]
        selected_outputs = {
            key: version.phase_outputs.get(key)
            for key in keys
            if key in version.phase_outputs
        }
        content = {
            "version_number": version.version_number,
            "status": version.status,
            "trigger_type": version.trigger_type,
            "rerun_from_phase": version.rerun_from_phase,
            "phase_outputs": selected_outputs,
        }
        if phase in {"claim_calculation", "claim_recommendation"}:
            content["recommendation"] = version.recommendation
        return ContextItem(
            source_type="assessment_version",
            source_id=version.assessment_version_id,
            claim_id=version.claim_id,
            related_phase=version.rerun_from_phase,
            timestamp=version.created_at,
            content=content,
            metadata={
                "parent_version_id": (
                    str(version.parent_version_id)
                    if version.parent_version_id is not None else None
                ),
                "trigger_reference_id": (
                    str(version.trigger_reference_id)
                    if version.trigger_reference_id is not None else None
                ),
                "completed_phases": version.completed_phases,
            },
        )

    @staticmethod
    def _ranked_bundle(
        phase: PhaseName,
        candidates: list[tuple[int, ContextItem]],
        max_items: int,
    ) -> ContextBundle:
        unique: dict[tuple[str, uuid.UUID], tuple[int, ContextItem]] = {}
        for candidate in candidates:
            item = candidate[1]
            unique.setdefault((item.source_type, item.source_id), candidate)
        ordered = sorted(
            unique.values(),
            key=lambda entry: (
                entry[0],
                -entry[1].timestamp.timestamp(),
                entry[1].source_type,
                str(entry[1].source_id),
            ),
        )
        truncated = len(ordered) > max_items
        items = [item for _, item in ordered[:max_items]]
        return ContextBundle(
            current_phase=phase,
            items=items,
            total_items=len(items),
            truncated=truncated,
        )

    @staticmethod
    def _requirement_phase(required_for: str) -> PhaseName | None:
        mapping: dict[str, PhaseName] = {
            "fact_validation": "fact_validation",
            "policy_applicability": "coverage_assessment",
            "coverage_assessment": "coverage_assessment",
            "exclusion_assessment": "exclusion_assessment",
            "obligation_assessment": "obligation_assessment",
            "claim_calculation": "claim_calculation",
        }
        return mapping.get(required_for)

    @staticmethod
    def _validate_limit(max_items: int) -> None:
        if not 1 <= max_items <= 50:
            raise ValueError("max_items must be between 1 and 50")

    @staticmethod
    def _validate_window(before: int, after: int) -> None:
        if not 0 <= before <= 20 or not 0 <= after <= 20:
            raise ValueError("before and after must be between 0 and 20")
