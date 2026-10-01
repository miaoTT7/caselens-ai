"""Lightweight insurance claim domain and AI assessment schemas."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PolicyReference(BaseModel):
    id: uuid.UUID
    policy_number: str | None = None
    policy_version: str | None = None
    product_code: str | None = None
    insured_party_reference: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    currency: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Incident(BaseModel):
    id: uuid.UUID
    incident_type: str | None = None
    event_date: datetime | None = None
    location: str | None = None
    cause: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Exposure(BaseModel):
    id: uuid.UUID
    incident_id: uuid.UUID
    coverage_reference: str
    exposure_type: str | None = None
    claimant_reference: str | None = None
    description: str | None = None
    claimed_amount: Decimal | None = None
    currency: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(BaseModel):
    id: uuid.UUID
    claim_number: str | None = None
    policy: PolicyReference | None = None
    reported_date: datetime | None = None
    claimant_reference: str | None = None
    description: str | None = None
    incidents: list[Incident] = Field(default_factory=list)
    exposures: list[Exposure] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceRef(BaseModel):
    id: str
    evidence_type: Literal["claim", "policy"]
    document_id: uuid.UUID | None = None
    chunk_id: uuid.UUID | None = None
    source_file: str | None = None
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    text_quote: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimFact(BaseModel):
    id: uuid.UUID
    fact_path: str
    value: Any | None
    unit: str | None = None
    status: Literal["reported", "extracted", "confirmed", "disputed", "unknown"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    claim_evidence: list[EvidenceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissingInformation(BaseModel):
    id: uuid.UUID
    field_path: str
    reason: str
    required_for: str
    question: str | None = None
    related_incident_id: uuid.UUID | None = None
    related_exposure_id: uuid.UUID | None = None
    blocking: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoverageAssessment(BaseModel):
    id: uuid.UUID
    # Candidate coverage can be assessed before an Exposure has been created.
    exposure_id: uuid.UUID | None = None
    coverage_reference: str
    status: Literal["covered", "potentially_covered", "not_covered", "indeterminate"]
    rationale: str | None = None
    matched_conditions: list[str] = Field(default_factory=list)
    unmatched_conditions: list[str] = Field(default_factory=list)
    unknown_conditions: list[str] = Field(default_factory=list)
    policy_evidence: list[EvidenceRef] = Field(default_factory=list)
    claim_evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information_ids: list[uuid.UUID] = Field(default_factory=list)


class ExclusionAssessment(BaseModel):
    id: uuid.UUID
    coverage_assessment_id: uuid.UUID
    exposure_id: uuid.UUID | None = None
    exclusion_reference: str
    status: Literal["applies", "does_not_apply", "indeterminate"]
    rationale: str | None = None
    matched_conditions: list[str] = Field(default_factory=list)
    unmatched_conditions: list[str] = Field(default_factory=list)
    unknown_conditions: list[str] = Field(default_factory=list)
    policy_evidence: list[EvidenceRef] = Field(default_factory=list)
    claim_evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information_ids: list[uuid.UUID] = Field(default_factory=list)


class ObligationAssessment(BaseModel):
    id: uuid.UUID
    coverage_assessment_id: uuid.UUID
    obligation_reference: str
    exposure_id: uuid.UUID | None = None
    status: Literal["satisfied", "breached", "indeterminate"]
    culpable_breach: bool | None = None
    effect_on_loss: Literal["affected", "not_affected", "unknown"]
    permitted_consequence: str | None = None
    rationale: str | None = None
    matched_conditions: list[str] = Field(default_factory=list)
    unmatched_conditions: list[str] = Field(default_factory=list)
    unknown_conditions: list[str] = Field(default_factory=list)
    policy_evidence: list[EvidenceRef] = Field(default_factory=list)
    claim_evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information_ids: list[uuid.UUID] = Field(default_factory=list)


class CalculationResult(BaseModel):
    id: uuid.UUID
    coverage_assessment_id: uuid.UUID
    exposure_id: uuid.UUID | None = None
    status: Literal["complete", "incomplete", "not_applicable"]
    currency: str | None = None
    claimed_amount: Decimal | None = None
    eligible_amount: Decimal | None = None
    deductible: Decimal | None = None
    limit: Decimal | None = None
    sublimit: Decimal | None = None
    other_insurance_amount: Decimal | None = None
    payable_amount: Decimal | None = None
    calculation_steps: list[str] = Field(default_factory=list)
    policy_evidence: list[EvidenceRef] = Field(default_factory=list)
    claim_evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information_ids: list[uuid.UUID] = Field(default_factory=list)


class ClaimRecommendation(BaseModel):
    id: uuid.UUID
    status: Literal[
        "recommend_approve",
        "recommend_partial",
        "recommend_decline",
        "needs_information",
        "needs_human_review",
        "no_recommendation",
    ]
    recommended_payable_amount: Decimal | None = None
    currency: str | None = None
    reasons: list[str] = Field(default_factory=list)
    supporting_assessment_ids: list[uuid.UUID] = Field(default_factory=list)
    human_review_required: bool
    disclaimer: str | None = None


class ClaimAssessment(BaseModel):
    id: uuid.UUID
    claim_id: uuid.UUID
    assessment_version: str
    status: Literal["incomplete", "completed", "needs_information", "needs_human_review"]
    facts: list[ClaimFact] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    recommendation: ClaimRecommendation | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimFactExtractionRequest(BaseModel):
    fnol_text: str | None = None
    parsed_document_text: str | None = None
    parsed_document_name: str | None = None

    @model_validator(mode="after")
    def require_input_text(self):
        if not (self.fnol_text and self.fnol_text.strip()) and not (
            self.parsed_document_text and self.parsed_document_text.strip()
        ):
            raise ValueError("FNOL text or parsed document text is required")
        return self


class ClaimFactExtractionResponse(BaseModel):
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    claim_evidence: list[EvidenceRef] = Field(default_factory=list)


class FactValidationRequest(BaseModel):
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    claimant_reference_required: bool = False
    policy_reference_required: bool = False


class FactValidationResponse(BaseModel):
    claim_id: uuid.UUID
    ready_for_assessment: bool
    missing_information: list[MissingInformation] = Field(default_factory=list)


class ApplicablePolicyAssessment(BaseModel):
    id: uuid.UUID
    status: Literal["identified", "candidate", "indeterminate"]
    policy: PolicyReference | None = None
    document_id: uuid.UUID | None = None
    source_file: str | None = None
    policy_evidence: list[EvidenceRef] = Field(default_factory=list)
    missing_information_ids: list[uuid.UUID] = Field(default_factory=list)


class CoverageAssessmentRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    retrieval_limit: int = Field(default=8, ge=1, le=20)


class CoverageAssessmentResponse(BaseModel):
    claim_id: uuid.UUID
    applicable_policy: ApplicablePolicyAssessment
    candidate_policy_sources: list[str] = Field(default_factory=list)
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)


class ExclusionAssessmentRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    retrieval_limit: int = Field(default=8, ge=1, le=20)


class ExclusionAssessmentResponse(BaseModel):
    claim_id: uuid.UUID
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)


class ObligationAssessmentRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    retrieval_limit: int = Field(default=8, ge=1, le=20)


class ObligationAssessmentResponse(BaseModel):
    claim_id: uuid.UUID
    status: Literal["completed", "unavailable"] = "completed"
    error: str | None = None
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)


class ClaimCalculationRequest(BaseModel):
    knowledge_base_id: uuid.UUID
    claim: Claim
    facts: list[ClaimFact] = Field(default_factory=list)
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    retrieval_limit: int = Field(default=8, ge=1, le=20)


class ClaimCalculationResponse(BaseModel):
    claim_id: uuid.UUID
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)


class ClaimRecommendationRequest(BaseModel):
    claim: Claim
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)


class ClaimRecommendationResponse(BaseModel):
    claim_id: uuid.UUID
    recommendation: ClaimRecommendation


class ClaimAssessmentRequest(BaseModel):
    fnol_text: str = Field(min_length=1)
    parsed_document_text: str | None = None
    parsed_document_name: str | None = None
    knowledge_base_id: uuid.UUID
    retrieval_limit: int = Field(default=8, ge=1, le=20)
    claimant_reference_required: bool = False
    policy_reference_required: bool = False


class ClaimAssessmentResponse(BaseModel):
    status: Literal["completed", "partial", "failed"]
    completed_phases: list[str] = Field(default_factory=list)
    failed_phase: str | None = None
    error: str | None = None
    claim: Claim | None = None
    facts: list[ClaimFact] = Field(default_factory=list)
    validation_result: FactValidationResponse | None = None
    missing_information: list[MissingInformation] = Field(default_factory=list)
    applicable_policy: ApplicablePolicyAssessment | None = None
    coverage_assessments: list[CoverageAssessment] = Field(default_factory=list)
    exclusion_assessments: list[ExclusionAssessment] = Field(default_factory=list)
    obligation_assessment_status: Literal["completed", "unavailable"] = "completed"
    obligation_assessment_error: str | None = None
    obligation_assessments: list[ObligationAssessment] = Field(default_factory=list)
    calculation_results: list[CalculationResult] = Field(default_factory=list)
    recommendation: ClaimRecommendation | None = None
