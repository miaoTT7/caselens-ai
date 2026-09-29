import unittest
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from pydantic import ValidationError

from api.claim_schemas import (
    CalculationResult,
    Claim,
    ClaimAssessment,
    ClaimFact,
    Exposure,
)


class ClaimSchemaTests(unittest.TestCase):
    def test_incomplete_claim_keeps_nullable_fields_and_empty_collections(self):
        claim = Claim(id=uuid.uuid4())

        self.assertIsNone(claim.policy)
        self.assertIsNone(claim.reported_date)
        self.assertEqual(claim.incidents, [])
        self.assertEqual(claim.exposures, [])

    def test_exposure_requires_incident_and_coverage_references(self):
        with self.assertRaises(ValidationError):
            Exposure(id=uuid.uuid4(), incident_id=uuid.uuid4())

    def test_nullable_fact_value_must_still_be_explicit(self):
        fact = ClaimFact(
            id=uuid.uuid4(),
            fact_path="incident.event_date",
            value=None,
            status="unknown",
        )
        self.assertIsNone(fact.value)

        with self.assertRaises(ValidationError):
            ClaimFact(id=uuid.uuid4(), fact_path="incident.event_date", status="unknown")

    def test_calculation_uses_decimal_and_allows_incomplete_amounts(self):
        calculation = CalculationResult(
            id=uuid.uuid4(),
            coverage_assessment_id=uuid.uuid4(),
            exposure_id=uuid.uuid4(),
            status="incomplete",
            claimed_amount=Decimal("1800.50"),
        )

        self.assertEqual(calculation.claimed_amount, Decimal("1800.50"))
        self.assertIsNone(calculation.payable_amount)

    def test_claim_assessment_defaults_to_empty_module_results(self):
        assessment = ClaimAssessment(
            id=uuid.uuid4(),
            claim_id=uuid.uuid4(),
            assessment_version="v1",
            status="incomplete",
            created_at=datetime.now(timezone.utc),
        )

        self.assertEqual(assessment.facts, [])
        self.assertEqual(assessment.coverage_assessments, [])
        self.assertEqual(assessment.calculation_results, [])
        self.assertIsNone(assessment.recommendation)


if __name__ == "__main__":
    unittest.main()
