import unittest
import uuid
from decimal import Decimal

from api.claim_recommendation import ClaimRecommendationService
from api.claim_schemas import (
    CalculationResult,
    Claim,
    ClaimRecommendationRequest,
    CoverageAssessment,
    ExclusionAssessment,
    MissingInformation,
    ObligationAssessment,
)


class ClaimRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.claim = Claim(id=uuid.uuid4())
        self.coverage = CoverageAssessment(
            id=uuid.uuid4(), coverage_reference="Contents", status="covered"
        )
        self.service = ClaimRecommendationService()

    def calculation(self, *, status="complete", payable="1000", eligible="1000", claimed="1000",
                    steps=None):
        return CalculationResult(
            id=uuid.uuid4(), coverage_assessment_id=self.coverage.id, status=status,
            currency="CHF", claimed_amount=Decimal(claimed), eligible_amount=Decimal(eligible),
            payable_amount=Decimal(payable) if payable is not None else None,
            calculation_steps=steps or [],
        )

    def recommend(self, **overrides):
        data = {
            "claim": self.claim,
            "coverage_assessments": [self.coverage],
            "calculation_results": [self.calculation()],
        }
        data.update(overrides)
        return self.service.recommend(ClaimRecommendationRequest(**data)).recommendation

    def test_clean_covered_claim_recommends_approve(self):
        result = self.recommend()
        self.assertEqual(result.status, "recommend_approve")
        self.assertEqual(result.recommended_payable_amount, Decimal("1000"))

    def test_supported_reduction_recommends_partial(self):
        calculation = self.calculation(
            payable="800", eligible="1000", steps=["Subtract deductible: CHF 1000 - CHF 200"]
        )
        result = self.recommend(calculation_results=[calculation])
        self.assertEqual(result.status, "recommend_partial")
        self.assertEqual(result.recommended_payable_amount, Decimal("800"))

    def test_not_covered_recommends_decline(self):
        coverage = CoverageAssessment(
            id=uuid.uuid4(), coverage_reference="Contents", status="not_covered"
        )
        result = self.recommend(coverage_assessments=[coverage], calculation_results=[])
        self.assertEqual(result.status, "recommend_decline")
        self.assertIsNone(result.recommended_payable_amount)

    def test_applying_exclusion_recommends_decline(self):
        exclusion = ExclusionAssessment(
            id=uuid.uuid4(), coverage_assessment_id=self.coverage.id,
            exclusion_reference="Intentional damage", status="applies",
        )
        result = self.recommend(exclusion_assessments=[exclusion])
        self.assertEqual(result.status, "recommend_decline")
        self.assertIn(exclusion.id, result.supporting_assessment_ids)

    def test_blocking_missing_information_takes_precedence(self):
        missing = MissingInformation(
            id=uuid.uuid4(), field_path="claim.receipt", reason="Receipt required",
            required_for="claim_calculation", blocking=True,
        )
        result = self.recommend(missing_information=[missing])
        self.assertEqual(result.status, "needs_information")
        self.assertIsNone(result.recommended_payable_amount)

    def test_unresolved_obligation_consequence_needs_human_review(self):
        obligation = ObligationAssessment(
            id=uuid.uuid4(), coverage_assessment_id=self.coverage.id,
            obligation_reference="Notify promptly", status="breached",
            effect_on_loss="unknown", permitted_consequence="Payment may be reduced.",
        )
        result = self.recommend(obligation_assessments=[obligation])
        self.assertEqual(result.status, "needs_human_review")
        self.assertTrue(result.human_review_required)

    def test_incomplete_calculation_needs_information(self):
        calculation = self.calculation(status="incomplete", payable=None)
        result = self.recommend(calculation_results=[calculation])
        self.assertEqual(result.status, "needs_information")
        self.assertIsNone(result.recommended_payable_amount)

    def test_payable_amount_is_copied_only_from_calculation(self):
        calculation = self.calculation(payable="750", eligible="1000", claimed="5000",
                                       steps=["Apply sublimit: CHF 1000 -> CHF 750"])
        result = self.recommend(calculation_results=[calculation])
        self.assertEqual(result.recommended_payable_amount, Decimal("750"))
        self.assertNotEqual(result.recommended_payable_amount, calculation.claimed_amount)

    def test_obligation_breach_alone_does_not_decline(self):
        obligation = ObligationAssessment(
            id=uuid.uuid4(), coverage_assessment_id=self.coverage.id,
            obligation_reference="Cooperate", status="breached", effect_on_loss="unknown",
        )
        result = self.recommend(obligation_assessments=[obligation])
        self.assertEqual(result.status, "needs_human_review")
        self.assertNotEqual(result.status, "recommend_decline")


if __name__ == "__main__":
    unittest.main()
