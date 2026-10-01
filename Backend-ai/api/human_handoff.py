"""Deterministic handoff packages for human claims reviewers."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from api.agent import CaseLensAgentState, PhaseName
from api.claim_schemas import (
    CalculationResult,
    ClaimRecommendation,
    CoverageAssessment,
    EvidenceRef,
    ExclusionAssessment,
    MissingInformation,
    ObligationAssessment,
)


HumanReviewReason = Literal[
    "provider_unavailable",
    "conflicting_assessment",
    "obligation_breach_requires_review",
    "recommendation_requires_human_review",
    "repeated_rerun_failure",
    "unresolved_assessment",
]


class HumanReviewHandoff(BaseModel):
    claim_id: uuid.UUID
    reason: HumanReviewReason
    current_phase: PhaseName | None = None
    completed_phases: list[PhaseName] = Field(default_factory=list)
    blocking_missing_information: list[MissingInformation] = Field(default_factory=list)
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    recommendation: ClaimRecommendation | None = None
    provider_errors: list[str] = Field(default_factory=list)
    relevant_evidence: list[EvidenceRef] = Field(default_factory=list)
    suggested_review_focus: list[str] = Field(default_factory=list)


class HumanReviewHandoffBuilder:
    def build(self, state: CaseLensAgentState) -> HumanReviewHandoff:
        if state.next_action != "human_review":
            raise ValueError("Human handoff requires next_action=human_review")
        if state.claim is None:
            raise ValueError("Human handoff requires a claim")

        reason = self._reason(state)
        blocking = [item for item in state.missing_information if item.blocking]
        return HumanReviewHandoff(
            claim_id=state.claim.id,
            reason=reason,
            current_phase=state.current_phase,
            completed_phases=state.completed_phases,
            blocking_missing_information=blocking,
            coverage_assessments=state.coverage_assessments,
            exclusion_assessments=state.exclusion_assessments,
            obligation_assessments=state.obligation_assessments,
            calculation_results=state.calculation_results,
            recommendation=state.recommendation,
            provider_errors=state.provider_errors,
            relevant_evidence=self._evidence(state),
            suggested_review_focus=self._focus(state, reason, blocking),
        )

    @staticmethod
    def _reason(state: CaseLensAgentState) -> HumanReviewReason:
        if state.rerun_failure_count >= 2:
            return "repeated_rerun_failure"
        if state.provider_errors or any(
            item.metadata.get("assessment_unavailable") is True
            for item in state.missing_information
        ):
            return "provider_unavailable"
        if state.recommendation and any(
            "conflict" in reason.casefold() for reason in state.recommendation.reasons
        ):
            return "conflicting_assessment"
        if any(item.status == "breached" for item in state.obligation_assessments):
            return "obligation_breach_requires_review"
        if state.recommendation and state.recommendation.status == "needs_human_review":
            return "recommendation_requires_human_review"
        return "unresolved_assessment"

    @staticmethod
    def _evidence(state: CaseLensAgentState) -> list[EvidenceRef]:
        candidates: list[EvidenceRef] = []
        if state.applicable_policy is not None:
            candidates.extend(state.applicable_policy.policy_evidence)
        for assessment in (
            *state.coverage_assessments,
            *state.exclusion_assessments,
            *state.obligation_assessments,
            *state.calculation_results,
        ):
            candidates.extend(getattr(assessment, "policy_evidence", []))
            candidates.extend(getattr(assessment, "claim_evidence", []))

        unique: dict[tuple, EvidenceRef] = {}
        for evidence in candidates:
            key = (
                evidence.evidence_type,
                evidence.id,
                evidence.document_id,
                evidence.chunk_id,
                evidence.text_quote,
            )
            unique.setdefault(key, evidence)
        return list(unique.values())

    @staticmethod
    def _focus(
        state: CaseLensAgentState,
        reason: HumanReviewReason,
        blocking: list[MissingInformation],
    ) -> list[str]:
        focus = {
            "provider_unavailable": "Complete the unavailable automated assessment manually.",
            "conflicting_assessment": "Resolve the conflicting structured assessment outcomes.",
            "obligation_breach_requires_review": (
                "Review the established obligation breach and any explicitly supported consequence."
            ),
            "recommendation_requires_human_review": (
                "Review the structured assessments supporting the human-review recommendation."
            ),
            "repeated_rerun_failure": "Review the phase that failed repeatedly and complete it manually.",
            "unresolved_assessment": "Resolve the remaining ambiguous structured assessments.",
        }[reason]
        result = [focus]
        if blocking:
            fields = ", ".join(dict.fromkeys(item.field_path for item in blocking))
            result.append(f"Resolve blocking information fields: {fields}.")
        breached = [
            item.obligation_reference
            for item in state.obligation_assessments
            if item.status == "breached"
        ]
        if breached:
            result.append(f"Review breached obligations: {', '.join(breached)}.")
        return result
