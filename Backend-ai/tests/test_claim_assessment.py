import unittest
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from api.claim_assessment import ClaimAssessmentOrchestrator
from api.claim_recommendation import ClaimRecommendationService
from api.claim_schemas import (
    ApplicablePolicyAssessment,
    CalculationResult,
    Claim,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimFact,
    ClaimFactExtractionResponse,
    ClaimRecommendation,
    ClaimRecommendationResponse,
    CoverageAssessment,
    CoverageAssessmentResponse,
    EvidenceRef,
    ExclusionAssessment,
    ExclusionAssessmentResponse,
    FactValidationResponse,
    Incident,
    MissingInformation,
    ObligationAssessment,
    ObligationAssessmentResponse,
)
from api.main import app, get_claim_assessment_orchestrator


class ClaimAssessmentOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    def build_services(self):
        claim = Claim(id=uuid.uuid4(), incidents=[Incident(id=uuid.uuid4(), incident_type="theft")])
        fact = ClaimFact(
            id=uuid.uuid4(), fact_path="incidents[0].incident_type",
            value="theft", status="confirmed",
        )
        document_id = uuid.uuid4()
        policy_evidence = EvidenceRef(
            id="[P1]", evidence_type="policy", document_id=document_id,
            source_file="policy.pdf", text_quote="Theft coverage.",
        )
        coverage = CoverageAssessment(
            id=uuid.uuid4(), coverage_reference="Theft", status="covered",
            policy_evidence=[policy_evidence],
        )
        exclusion = ExclusionAssessment(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            exclusion_reference="Unattended property", status="does_not_apply",
        )
        obligation = ObligationAssessment(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            obligation_reference="Police report", status="satisfied", effect_on_loss="unknown",
        )
        calculation = CalculationResult(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id, status="complete",
            currency="CHF", claimed_amount=Decimal("1000"), eligible_amount=Decimal("1000"),
            payable_amount=Decimal("1000"),
        )
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(), status="recommend_approve",
            recommended_payable_amount=Decimal("1000"), currency="CHF",
            reasons=["Structured results support payment."],
            supporting_assessment_ids=[coverage.id, calculation.id],
            human_review_required=False, disclaimer="Decision support only.",
        )
        duplicate_one = MissingInformation(
            id=uuid.uuid4(), field_path="claim.optional_note", reason="Optional note missing",
            required_for="assessment", blocking=False,
        )
        duplicate_two = duplicate_one.model_copy(update={"id": uuid.uuid4()})

        extraction = Mock()
        extraction.extract = AsyncMock(return_value=ClaimFactExtractionResponse(
            claim=claim, facts=[fact], claim_evidence=[]
        ))
        validation = Mock()
        validation.validate = Mock(return_value=FactValidationResponse(
            claim_id=claim.id, ready_for_assessment=True,
            missing_information=[duplicate_one],
        ))
        coverage_service = Mock()
        coverage_service.assess = AsyncMock(return_value=CoverageAssessmentResponse(
            claim_id=claim.id,
            applicable_policy=ApplicablePolicyAssessment(
                id=uuid.uuid4(), status="identified", document_id=document_id,
            ),
            coverage_assessments=[coverage], missing_information=[duplicate_two],
        ))
        exclusion_service = Mock()
        exclusion_service.assess = AsyncMock(return_value=ExclusionAssessmentResponse(
            claim_id=claim.id, exclusion_assessments=[exclusion]
        ))
        obligation_service = Mock()
        obligation_service.assess = AsyncMock(return_value=ObligationAssessmentResponse(
            claim_id=claim.id, obligation_assessments=[obligation]
        ))
        calculation_service = Mock()
        calculation_service.calculate = AsyncMock(return_value=SimpleCalculationResponse(
            claim_id=claim.id, calculation_results=[calculation], missing_information=[]
        ))
        recommendation_service = Mock()
        recommendation_service.recommend = Mock(return_value=ClaimRecommendationResponse(
            claim_id=claim.id, recommendation=recommendation
        ))
        services = {
            "extraction_service": extraction,
            "validation_service": validation,
            "coverage_service": coverage_service,
            "exclusion_service": exclusion_service,
            "obligation_service": obligation_service,
            "calculation_service": calculation_service,
            "recommendation_service": recommendation_service,
        }
        return services, claim, fact, coverage, exclusion, obligation, calculation

    async def test_full_workflow_passes_structured_outputs_between_phases(self):
        services, claim, fact, coverage, exclusion, obligation, calculation = self.build_services()
        orchestrator = ClaimAssessmentOrchestrator(**services)
        result = await orchestrator.assess(ClaimAssessmentRequest(
            fnol_text="My bicycle was stolen.", knowledge_base_id=uuid.uuid4(), retrieval_limit=6
        ))

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.completed_phases), 7)
        self.assertEqual(result.claim.id, claim.id)
        self.assertEqual(result.recommendation.status, "recommend_approve")
        self.assertEqual(len(result.missing_information), 1)

        coverage_request = services["coverage_service"].assess.await_args.args[0]
        self.assertEqual(coverage_request.facts[0].id, fact.id)
        exclusion_request = services["exclusion_service"].assess.await_args.args[0]
        self.assertEqual(exclusion_request.coverage_assessments[0].id, coverage.id)
        obligation_request = services["obligation_service"].assess.await_args.args[0]
        self.assertEqual(obligation_request.exclusion_assessments[0].id, exclusion.id)
        calculation_request = services["calculation_service"].calculate.await_args.args[0]
        self.assertEqual(calculation_request.obligation_assessments[0].id, obligation.id)
        recommendation_request = services["recommendation_service"].recommend.call_args.args[0]
        self.assertEqual(recommendation_request.calculation_results[0].id, calculation.id)

    async def test_blocking_validation_returns_partial_and_stops_dependents(self):
        services, claim, _, _, _, _, _ = self.build_services()
        services["validation_service"].validate.return_value = FactValidationResponse(
            claim_id=claim.id, ready_for_assessment=False,
            missing_information=[MissingInformation(
                id=uuid.uuid4(), field_path="incidents[0].event_date",
                reason="Event date required", required_for="fact_validation", blocking=True,
            )],
        )
        result = await ClaimAssessmentOrchestrator(**services).assess(ClaimAssessmentRequest(
            fnol_text="My bicycle was stolen.", knowledge_base_id=uuid.uuid4()
        ))

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.failed_phase, "coverage_assessment")
        services["coverage_service"].assess.assert_not_awaited()
        services["recommendation_service"].recommend.assert_not_called()

    async def test_unavailable_obligations_continue_and_require_human_review(self):
        services, claim, _, _, _, _, _ = self.build_services()
        unavailable = MissingInformation(
            id=uuid.uuid4(),
            field_path="obligations.provider_assessment",
            reason="Obligation assessment provider unavailable.",
            required_for="obligation_assessment",
            blocking=True,
            metadata={"assessment_unavailable": True},
        )
        services["obligation_service"].assess.return_value = ObligationAssessmentResponse(
            claim_id=claim.id,
            status="unavailable",
            error="Obligation assessment is unavailable due to an LLM provider failure.",
            missing_information=[unavailable],
        )
        services["recommendation_service"] = ClaimRecommendationService()

        result = await ClaimAssessmentOrchestrator(**services).assess(ClaimAssessmentRequest(
            fnol_text="My bicycle was stolen.", knowledge_base_id=uuid.uuid4()
        ))

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.obligation_assessment_status, "unavailable")
        self.assertEqual(result.obligation_assessments, [])
        self.assertEqual(result.recommendation.status, "needs_human_review")
        self.assertTrue(result.recommendation.human_review_required)
        services["calculation_service"].calculate.assert_awaited_once()


class SimpleCalculationResponse:
    def __init__(self, claim_id, calculation_results, missing_information):
        self.claim_id = claim_id
        self.calculation_results = calculation_results
        self.missing_information = missing_information


class ClaimAssessmentEndpointTests(unittest.TestCase):
    def test_one_post_returns_full_result_without_get(self):
        claim_id = uuid.uuid4()
        recommendation = ClaimRecommendation(
            id=uuid.uuid4(), status="no_recommendation", reasons=["No assessment."],
            human_review_required=False, disclaimer="Decision support only.",
        )
        orchestrator = Mock()
        orchestrator.assess = AsyncMock(return_value=ClaimAssessmentResponse(
            status="completed", completed_phases=["claim_recommendation"],
            claim=Claim(id=claim_id), recommendation=recommendation,
        ))
        app.dependency_overrides[get_claim_assessment_orchestrator] = lambda: orchestrator
        try:
            response = TestClient(app).post("/claims/assess", json={
                "fnol_text": "A loss occurred.",
                "knowledge_base_id": str(uuid.uuid4()),
            })
        finally:
            app.dependency_overrides.pop(get_claim_assessment_orchestrator, None)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["claim"]["id"], str(claim_id))
        self.assertEqual(response.json()["recommendation"]["status"], "no_recommendation")
        orchestrator.assess.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
