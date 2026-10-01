"""Immutable historical snapshots for stable claim assessment outcomes."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from api.agent import CaseLensAgentState, PhaseName
from api.claim_schemas import (
    ApplicablePolicyAssessment,
    CalculationResult,
    Claim,
    ClaimFact,
    ClaimRecommendation,
    CoverageAssessment,
    ExclusionAssessment,
    MissingInformation,
    ObligationAssessment,
)
from api.models import AssessmentVersionRecord
from api.repositories import AssessmentVersionRepository


AssessmentTriggerType = Literal[
    "initial_assessment",
    "user_follow_up",
    "selective_rerun",
    "provider_retry",
    "manual_review",
]
AssessmentVersionStatus = Literal[
    "incomplete",
    "completed",
    "needs_information",
    "needs_human_review",
    "failed",
]


class AssessmentPhaseOutputs(BaseModel):
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    applicable_policy: ApplicablePolicyAssessment | None = None
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    provider_errors: list[str] = Field(default_factory=list)
    rerun_failure_count: int = Field(ge=0)
    current_phase: PhaseName | None = None
    next_action: str


class AssessmentVersionPersistenceError(RuntimeError):
    pass


class AssessmentVersionService:
    def __init__(self, repository: AssessmentVersionRepository):
        self.repository = repository

    async def create_version(
        self,
        state: CaseLensAgentState,
        *,
        trigger_type: AssessmentTriggerType,
        trigger_reference_id: uuid.UUID | None = None,
        rerun_from_phase: PhaseName | None = None,
    ) -> AssessmentVersionRecord:
        if state.claim is None:
            raise ValueError("Assessment version requires a claim")
        if not self.is_stable_assessment(state):
            raise ValueError("Assessment state does not contain a stable structured result")
        if trigger_type == "initial_assessment" and rerun_from_phase is not None:
            raise ValueError("Initial assessment cannot have rerun_from_phase")

        claim_id = state.claim.id
        await self.repository.acquire_claim_lock(claim_id)
        parent = await self.repository.get_latest_version(claim_id)
        if trigger_type == "initial_assessment" and parent is not None:
            raise ValueError("Initial assessment version already exists")
        if trigger_type != "initial_assessment" and parent is None:
            raise ValueError("A non-initial assessment version requires a parent")

        phase_outputs = self._phase_outputs(state)
        recommendation = (
            state.recommendation.model_dump(mode="json")
            if state.recommendation is not None
            else None
        )
        try:
            return await self.repository.create_version(
                claim_id=claim_id,
                parent_version_id=(parent.assessment_version_id if parent else None),
                version_number=(parent.version_number + 1 if parent else 1),
                trigger_type=trigger_type,
                trigger_reference_id=trigger_reference_id,
                rerun_from_phase=rerun_from_phase,
                completed_phases=list(state.completed_phases),
                phase_outputs=phase_outputs.model_dump(mode="json"),
                recommendation=recommendation,
                status=self._status(state),
            )
        except Exception as error:
            raise AssessmentVersionPersistenceError(
                f"Failed to persist assessment version for claim {claim_id}"
            ) from error

    async def get_latest_version(
        self, claim_id: uuid.UUID
    ) -> AssessmentVersionRecord | None:
        return await self.repository.get_latest_version(claim_id)

    async def list_versions(
        self, claim_id: uuid.UUID
    ) -> Sequence[AssessmentVersionRecord]:
        return await self.repository.list_versions(claim_id)

    async def get_version(
        self, version_id: uuid.UUID
    ) -> AssessmentVersionRecord | None:
        return await self.repository.get_version(version_id)

    @staticmethod
    def has_structured_output(state: CaseLensAgentState) -> bool:
        return bool(
            state.applicable_policy is not None
            or state.coverage_assessments
            or state.exclusion_assessments
            or state.obligation_assessments
            or state.calculation_results
            or state.recommendation is not None
        )

    @classmethod
    def is_stable_assessment(cls, state: CaseLensAgentState) -> bool:
        return cls.has_structured_output(state) and state.next_action != "rerun_phase"

    @staticmethod
    def _phase_outputs(state: CaseLensAgentState) -> AssessmentPhaseOutputs:
        if state.claim is None:
            raise ValueError("Assessment version requires a claim")
        return AssessmentPhaseOutputs(
            claim=state.claim,
            facts=state.facts,
            missing_information=state.missing_information,
            applicable_policy=state.applicable_policy,
            coverage_assessments=state.coverage_assessments,
            exclusion_assessments=state.exclusion_assessments,
            obligation_assessments=state.obligation_assessments,
            calculation_results=state.calculation_results,
            provider_errors=state.provider_errors,
            rerun_failure_count=state.rerun_failure_count,
            current_phase=state.current_phase,
            next_action=state.next_action,
        )

    @staticmethod
    def _status(state: CaseLensAgentState) -> AssessmentVersionStatus:
        if state.next_action == "human_review":
            return "needs_human_review"
        if any(item.blocking for item in state.missing_information):
            return "needs_information"
        if state.next_action == "completed":
            return "completed"
        return "incomplete"
