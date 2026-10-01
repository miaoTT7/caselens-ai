import unittest
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.agent import (
    CaseLensAgent,
    CaseLensAgentRouter,
    CaseLensAgentState,
    MissingInformationAnswer,
    MissingInformationAnswerRequest,
    MissingInformationInteraction,
    SelectiveRerunner,
)
from api.claim_recommendation import ClaimRecommendationService
from api.claim_schemas import (
    ApplicablePolicyAssessment,
    CalculationResult,
    Claim,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
    ClaimFact,
    ClaimRecommendation,
    CoverageAssessment,
    ExclusionAssessment,
    ExclusionAssessmentResponse,
    FactValidationResponse,
    MissingInformation,
    ObligationAssessment,
    ObligationAssessmentResponse,
)


class CaseLensAgentRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = CaseLensAgentRouter()

    def test_blocking_information_routes_to_question(self):
        state = CaseLensAgentState(
            current_phase="fact_validation",
            missing_information=[MissingInformation(
                id=uuid.uuid4(),
                field_path="incidents[0].event_date",
                reason="Event date is required.",
                required_for="fact_validation",
                blocking=True,
            )],
        )

        self.assertEqual(self.router.route(state).next_action, "ask_for_information")

    def test_explicit_failed_phase_routes_to_rerun(self):
        response = ClaimAssessmentResponse(
            status="partial",
            failed_phase="exclusion_assessment",
            error="LLMProviderError: phase execution failed",
            claim=Claim(id=uuid.uuid4()),
        )

        state = self.router.route(CaseLensAgentState.from_assessment(response))

        self.assertEqual(state.current_phase, "exclusion_assessment")
        self.assertEqual(state.next_action, "rerun_phase")

    def test_unavailable_assessment_routes_to_human_review(self):
        state = CaseLensAgentState(missing_information=[MissingInformation(
            id=uuid.uuid4(),
            field_path="obligations.provider_assessment",
            reason="Provider unavailable.",
            required_for="obligation_assessment",
            blocking=True,
            metadata={"assessment_unavailable": True},
        )])

        self.assertEqual(self.router.route(state).next_action, "human_review")

    def test_final_recommendation_routes_to_completed_or_review(self):
        completed = CaseLensAgentState(recommendation=ClaimRecommendation(
            id=uuid.uuid4(),
            status="recommend_approve",
            reasons=["Assessment complete."],
            human_review_required=False,
            disclaimer="Decision support only.",
        ))
        review = completed.model_copy(update={"recommendation": ClaimRecommendation(
            id=uuid.uuid4(),
            status="needs_human_review",
            reasons=["Review required."],
            human_review_required=True,
            disclaimer="Decision support only.",
        )})

        self.assertEqual(self.router.route(completed).next_action, "completed")
        self.assertEqual(self.router.route(review).next_action, "human_review")


class CaseLensAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_reuses_existing_orchestrator(self):
        response = ClaimAssessmentResponse(
            status="completed",
            completed_phases=["claim_recommendation"],
            claim=Claim(id=uuid.uuid4()),
            recommendation=ClaimRecommendation(
                id=uuid.uuid4(),
                status="recommend_approve",
                reasons=["Assessment complete."],
                human_review_required=False,
                disclaimer="Decision support only.",
            ),
        )
        orchestrator = AsyncMock()
        orchestrator.assess.return_value = response
        request = ClaimAssessmentRequest(
            fnol_text="My bicycle was stolen.",
            knowledge_base_id=uuid.uuid4(),
        )

        state = await CaseLensAgent(orchestrator).assess(request)

        orchestrator.assess.assert_awaited_once_with(request)
        self.assertEqual(state.next_action, "completed")
        self.assertEqual(state.claim.id, response.claim.id)


class MissingInformationInteractionTests(unittest.TestCase):
    def setUp(self):
        self.incident_id = uuid.uuid4()
        self.location = MissingInformation(
            id=uuid.uuid4(),
            field_path="incidents[0].location",
            reason="Location is required for exclusion assessment.",
            required_for="exclusion_assessment",
            question="Where did the theft occur?",
            related_incident_id=self.incident_id,
            blocking=True,
        )
        self.amount = MissingInformation(
            id=uuid.uuid4(),
            field_path="claim.eligible_amount",
            reason="Eligible amount is required for calculation.",
            required_for="claim_calculation",
            question="What is the eligible loss amount?",
            blocking=True,
        )
        self.state = CaseLensAgentState(
            missing_information=[self.location, self.amount],
            next_action="ask_for_information",
        )

    def test_exposes_blocking_questions_with_routing_metadata(self):
        questions = MissingInformationInteraction.questions(self.state)

        self.assertEqual([item.id for item in questions], [self.location.id, self.amount.id])
        self.assertEqual(questions[0].field_path, "incidents[0].location")
        self.assertEqual(questions[0].required_for, "exclusion_assessment")
        self.assertEqual(questions[0].related_incident_id, self.incident_id)

    def test_answers_create_confirmed_facts_and_choose_earliest_phase(self):
        updated = MissingInformationInteraction.apply_answers(
            self.state,
            MissingInformationAnswerRequest(answers=[
                MissingInformationAnswer(
                    missing_information_id=self.amount.id,
                    value="1800",
                    original_text="The eligible loss is CHF 1800.",
                ),
                MissingInformationAnswer(
                    missing_information_id=self.location.id,
                    value="outside Zurich station",
                ),
            ]),
        )

        facts = {item.fact_path: item for item in updated.facts}
        self.assertEqual(facts["claim.eligible_amount"].value, "1800")
        self.assertEqual(facts["incidents[0].location"].status, "confirmed")
        self.assertEqual(
            facts["incidents[0].location"].metadata["related_incident_id"],
            str(self.incident_id),
        )
        self.assertEqual(updated.missing_information, [])
        self.assertEqual(updated.current_phase, "exclusion_assessment")
        self.assertEqual(updated.next_action, "rerun_phase")

    def test_answer_replaces_existing_fact_for_same_path(self):
        state = self.state.model_copy(update={"facts": [
            ClaimFact(
                id=uuid.uuid4(),
                fact_path="incidents[0].location",
                value="unknown station",
                status="reported",
            )
        ]})

        updated = MissingInformationInteraction.apply_answers(
            state,
            MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
                missing_information_id=self.location.id,
                value="outside Zurich station",
            )]),
        )

        matching = [
            item for item in updated.facts if item.fact_path == "incidents[0].location"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].value, "outside Zurich station")

    def test_unknown_answer_id_is_rejected_without_updating_state(self):
        with self.assertRaises(ValueError):
            MissingInformationInteraction.apply_answers(
                self.state,
                MissingInformationAnswerRequest(answers=[MissingInformationAnswer(
                    missing_information_id=uuid.uuid4(),
                    value="unsupported",
                )]),
            )

        self.assertEqual(self.state.facts, [])


class SelectiveRerunnerTests(unittest.IsolatedAsyncioTestCase):
    def build_rerunner(self):
        validation = SimpleNamespace(validate=AsyncMock())
        coverage = SimpleNamespace(assess=AsyncMock())
        exclusion = SimpleNamespace(assess=AsyncMock())
        obligation = SimpleNamespace(assess=AsyncMock())
        calculation = SimpleNamespace(calculate=AsyncMock())
        rerunner = SelectiveRerunner(
            validation_service=validation,
            coverage_service=coverage,
            exclusion_service=exclusion,
            obligation_service=obligation,
            calculation_service=calculation,
            recommendation_service=ClaimRecommendationService(),
        )
        return rerunner, validation, coverage, exclusion, obligation, calculation

    def base_state(self):
        claim = Claim(id=uuid.uuid4())
        coverage = CoverageAssessment(
            id=uuid.uuid4(), coverage_reference="Theft", status="covered"
        )
        exclusion = ExclusionAssessment(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            exclusion_reference="Old exclusion", status="does_not_apply",
        )
        obligation = ObligationAssessment(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            obligation_reference="Old obligation", status="satisfied",
            effect_on_loss="unknown",
        )
        calculation = CalculationResult(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            status="complete", currency="CHF",
            claimed_amount=Decimal("1000"), eligible_amount=Decimal("1000"),
            payable_amount=Decimal("1000"),
        )
        return CaseLensAgentState(
            knowledge_base_id=uuid.uuid4(),
            claim=claim,
            applicable_policy=ApplicablePolicyAssessment(
                id=uuid.uuid4(), status="identified"
            ),
            coverage_assessments=[coverage],
            exclusion_assessments=[exclusion],
            obligation_assessments=[obligation],
            calculation_results=[calculation],
            current_phase="exclusion_assessment",
            next_action="rerun_phase",
        )

    async def test_reruns_target_and_downstream_without_earlier_services(self):
        rerunner, validation, coverage_service, exclusion_service, obligation_service, calculation_service = (
            self.build_rerunner()
        )
        state = self.base_state()
        preserved_coverage = state.coverage_assessments[0]
        new_exclusion = ExclusionAssessment(
            id=uuid.uuid4(), coverage_assessment_id=preserved_coverage.id,
            exclusion_reference="New exclusion", status="does_not_apply",
        )
        new_obligation = ObligationAssessment(
            id=uuid.uuid4(), coverage_assessment_id=preserved_coverage.id,
            obligation_reference="New obligation", status="satisfied",
            effect_on_loss="unknown",
        )
        new_calculation = CalculationResult(
            id=uuid.uuid4(), coverage_assessment_id=preserved_coverage.id,
            status="complete", currency="CHF",
            claimed_amount=Decimal("1000"), eligible_amount=Decimal("1000"),
            payable_amount=Decimal("1000"),
        )
        exclusion_service.assess.return_value = ExclusionAssessmentResponse(
            claim_id=state.claim.id, exclusion_assessments=[new_exclusion]
        )
        obligation_service.assess.return_value = ObligationAssessmentResponse(
            claim_id=state.claim.id, obligation_assessments=[new_obligation]
        )
        calculation_service.calculate.return_value = SimpleNamespace(
            calculation_results=[new_calculation], missing_information=[]
        )

        result = await rerunner.rerun(state)

        validation.validate.assert_not_called()
        coverage_service.assess.assert_not_awaited()
        exclusion_service.assess.assert_awaited_once()
        obligation_service.assess.assert_awaited_once()
        calculation_service.calculate.assert_awaited_once()
        self.assertEqual(result.coverage_assessments[0].id, preserved_coverage.id)
        self.assertEqual(result.exclusion_assessments[0].id, new_exclusion.id)
        self.assertEqual(result.obligation_assessments[0].id, new_obligation.id)
        self.assertEqual(result.calculation_results[0].id, new_calculation.id)
        self.assertEqual(result.current_phase, "claim_recommendation")
        self.assertEqual(result.next_action, "completed")

    async def test_removes_only_stale_downstream_missing_information(self):
        rerunner, _, _, exclusion_service, obligation_service, calculation_service = (
            self.build_rerunner()
        )
        state = self.base_state()
        coverage_missing = MissingInformation(
            id=uuid.uuid4(), field_path="claim.policy.version",
            reason="Earlier coverage note", required_for="coverage_assessment", blocking=False,
        )
        stale_exclusion = MissingInformation(
            id=uuid.uuid4(), field_path="incidents[0].location",
            reason="Old exclusion question", required_for="exclusion_assessment", blocking=True,
        )
        state.missing_information = [coverage_missing, stale_exclusion]
        exclusion_service.assess.return_value = ExclusionAssessmentResponse(
            claim_id=state.claim.id, exclusion_assessments=[]
        )
        obligation_service.assess.return_value = ObligationAssessmentResponse(
            claim_id=state.claim.id, obligation_assessments=[]
        )
        calculation_service.calculate.return_value = SimpleNamespace(
            calculation_results=[], missing_information=[]
        )

        result = await rerunner.rerun(state)

        self.assertEqual([item.id for item in result.missing_information], [coverage_missing.id])

    async def test_failed_rerun_preserves_earlier_state_and_clears_stale_downstream(self):
        rerunner, _, _, exclusion_service, obligation_service, calculation_service = (
            self.build_rerunner()
        )
        state = self.base_state()
        preserved_coverage_id = state.coverage_assessments[0].id
        exclusion_service.assess.side_effect = RuntimeError("provider failed")

        result = await rerunner.rerun(state)

        self.assertEqual(result.coverage_assessments[0].id, preserved_coverage_id)
        self.assertEqual(result.exclusion_assessments, [])
        self.assertEqual(result.obligation_assessments, [])
        self.assertEqual(result.calculation_results, [])
        self.assertIsNone(result.recommendation)
        self.assertEqual(result.current_phase, "exclusion_assessment")
        self.assertEqual(result.next_action, "rerun_phase")
        obligation_service.assess.assert_not_awaited()
        calculation_service.calculate.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
