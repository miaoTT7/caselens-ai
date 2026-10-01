"""Thin orchestration layer for the existing Phase 1–7 services."""

from __future__ import annotations

from api.claim_calculation import ClaimCalculationService
from api.claim_fact_extraction import ClaimFactExtractionService
from api.claim_fact_validation import ClaimFactValidationService
from api.claim_recommendation import ClaimRecommendationService
from api.claim_schemas import (
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimCalculationRequest,
    ClaimFactExtractionRequest,
    ClaimRecommendationRequest,
    CoverageAssessmentRequest,
    ExclusionAssessmentRequest,
    FactValidationRequest,
    MissingInformation,
    ObligationAssessmentRequest,
)
from api.coverage_assessment import CoverageAssessmentService
from api.exclusion_assessment import ExclusionAssessmentService
from api.obligation_assessment import ObligationAssessmentService


class ClaimAssessmentOrchestrator:
    def __init__(
        self,
        session=None,
        embedding_model=None,
        *,
        extraction_service=None,
        validation_service=None,
        coverage_service=None,
        exclusion_service=None,
        obligation_service=None,
        calculation_service=None,
        recommendation_service=None,
    ):
        self.extraction = extraction_service or ClaimFactExtractionService()
        self.validation = validation_service or ClaimFactValidationService()
        self.coverage = coverage_service or CoverageAssessmentService(session, embedding_model)
        self.exclusion = exclusion_service or ExclusionAssessmentService(session, embedding_model)
        self.obligation = obligation_service or ObligationAssessmentService(session, embedding_model)
        self.calculation = calculation_service or ClaimCalculationService(session, embedding_model)
        self.recommendation = recommendation_service or ClaimRecommendationService()

    async def assess(self, request: ClaimAssessmentRequest) -> ClaimAssessmentResponse:
        state = ClaimAssessmentResponse(status="failed")
        current_phase = "claim_facts_extraction"
        try:
            extracted = await self.extraction.extract(ClaimFactExtractionRequest(
                fnol_text=request.fnol_text,
                parsed_document_text=request.parsed_document_text,
                parsed_document_name=request.parsed_document_name,
            ))
            state.claim = extracted.claim
            state.facts = extracted.facts
            state.completed_phases.append(current_phase)

            current_phase = "fact_validation"
            validation = self.validation.validate(FactValidationRequest(
                claim=extracted.claim,
                facts=extracted.facts,
                claimant_reference_required=request.claimant_reference_required,
                policy_reference_required=request.policy_reference_required,
            ))
            state.validation_result = validation
            state.missing_information.extend(validation.missing_information)
            state.completed_phases.append(current_phase)
            if not validation.ready_for_assessment:
                state.status = "partial"
                state.failed_phase = "coverage_assessment"
                state.error = "Blocking validation information is required before policy assessment."
                self._deduplicate_and_relink(state)
                return state

            current_phase = "coverage_assessment"
            coverage = await self.coverage.assess(CoverageAssessmentRequest(
                knowledge_base_id=request.knowledge_base_id,
                claim=extracted.claim,
                facts=extracted.facts,
                retrieval_limit=request.retrieval_limit,
            ))
            state.applicable_policy = coverage.applicable_policy
            state.coverage_assessments = coverage.coverage_assessments
            state.missing_information.extend(coverage.missing_information)
            state.completed_phases.append(current_phase)
            if not coverage.coverage_assessments:
                current_phase = "claim_recommendation"
                recommendation = self.recommendation.recommend(ClaimRecommendationRequest(
                    claim=extracted.claim,
                    missing_information=state.missing_information,
                ))
                state.recommendation = recommendation.recommendation
                state.completed_phases.append(current_phase)
                state.status = "partial"
                state.failed_phase = "exclusion_assessment"
                state.error = "No coverage assessment is available for dependent phases."
                self._deduplicate_and_relink(state)
                return state

            current_phase = "exclusion_assessment"
            exclusions = await self.exclusion.assess(ExclusionAssessmentRequest(
                knowledge_base_id=request.knowledge_base_id,
                claim=extracted.claim,
                facts=extracted.facts,
                coverage_assessments=coverage.coverage_assessments,
                retrieval_limit=request.retrieval_limit,
            ))
            state.exclusion_assessments = exclusions.exclusion_assessments
            state.missing_information.extend(exclusions.missing_information)
            state.completed_phases.append(current_phase)

            current_phase = "obligation_assessment"
            obligations = await self.obligation.assess(ObligationAssessmentRequest(
                knowledge_base_id=request.knowledge_base_id,
                claim=extracted.claim,
                facts=extracted.facts,
                coverage_assessments=coverage.coverage_assessments,
                exclusion_assessments=exclusions.exclusion_assessments,
                retrieval_limit=request.retrieval_limit,
            ))
            state.obligation_assessment_status = obligations.status
            state.obligation_assessment_error = obligations.error
            state.obligation_assessments = obligations.obligation_assessments
            state.missing_information.extend(obligations.missing_information)
            state.completed_phases.append(current_phase)

            current_phase = "claim_calculation"
            calculations = await self.calculation.calculate(ClaimCalculationRequest(
                knowledge_base_id=request.knowledge_base_id,
                claim=extracted.claim,
                facts=extracted.facts,
                coverage_assessments=coverage.coverage_assessments,
                exclusion_assessments=exclusions.exclusion_assessments,
                obligation_assessments=obligations.obligation_assessments,
                retrieval_limit=request.retrieval_limit,
            ))
            state.calculation_results = calculations.calculation_results
            state.missing_information.extend(calculations.missing_information)
            state.completed_phases.append(current_phase)

            self._deduplicate_and_relink(state)
            current_phase = "claim_recommendation"
            recommendation = self.recommendation.recommend(ClaimRecommendationRequest(
                claim=extracted.claim,
                coverage_assessments=coverage.coverage_assessments,
                exclusion_assessments=exclusions.exclusion_assessments,
                obligation_assessments=obligations.obligation_assessments,
                calculation_results=calculations.calculation_results,
                missing_information=state.missing_information,
            ))
            state.recommendation = recommendation.recommendation
            state.completed_phases.append(current_phase)
            state.status = "completed"
            return state
        except Exception as error:
            state.status = "failed" if state.claim is None else "partial"
            state.failed_phase = current_phase
            state.error = f"{type(error).__name__}: phase execution failed"
            self._deduplicate_and_relink(state)
            return state

    @classmethod
    def _deduplicate_and_relink(cls, state: ClaimAssessmentResponse) -> None:
        canonical: dict[tuple, MissingInformation] = {}
        aliases = {}
        merged = []
        for item in state.missing_information:
            key = (
                item.field_path, item.reason, item.required_for, item.blocking, item.question,
                item.related_incident_id, item.related_exposure_id,
            )
            existing = canonical.get(key)
            if existing is None:
                canonical[key] = item
                merged.append(item)
            else:
                aliases[item.id] = existing.id
        state.missing_information = merged

        objects = [
            state.applicable_policy,
            *state.coverage_assessments,
            *state.exclusion_assessments,
            *state.obligation_assessments,
            *state.calculation_results,
        ]
        for item in objects:
            if item is not None and hasattr(item, "missing_information_ids"):
                item.missing_information_ids = list(dict.fromkeys(
                    aliases.get(value, value) for value in item.missing_information_ids
                ))
