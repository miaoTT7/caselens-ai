import unittest
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from api.claim_fact_validation import ClaimFactValidationService
from api.claim_schemas import Claim, ClaimFact, FactValidationRequest, Incident


class ClaimFactValidationTests(unittest.TestCase):
    def setUp(self):
        self.service = ClaimFactValidationService()

    def test_missing_core_incident_fields_are_blocking(self):
        claim = Claim(id=uuid.uuid4(), incidents=[Incident(id=uuid.uuid4())])

        result = self.service.validate(FactValidationRequest(claim=claim))

        paths = {item.field_path: item for item in result.missing_information}
        self.assertTrue(paths["claim.incidents[0].incident_type"].blocking)
        self.assertTrue(paths["claim.incidents[0].event_date"].blocking)
        self.assertTrue(paths["claim.incidents[0].location"].blocking)
        self.assertFalse(paths["claim.claimed_amount"].blocking)
        self.assertNotIn("claim.currency", paths)
        self.assertFalse(result.ready_for_assessment)

    def test_amount_and_currency_are_non_blocking(self):
        incident = Incident(
            id=uuid.uuid4(),
            incident_type="theft",
            event_date=datetime(2026, 8, 12, tzinfo=timezone.utc),
            location="Zurich station",
        )
        amount = ClaimFact(
            id=uuid.uuid4(),
            fact_path="claim.claimed_amount",
            value=Decimal("1800"),
            status="extracted",
            confidence=0.9,
        )

        result = self.service.validate(
            FactValidationRequest(claim=Claim(id=uuid.uuid4(), incidents=[incident]), facts=[amount])
        )

        self.assertTrue(result.ready_for_assessment)
        self.assertEqual(len(result.missing_information), 1)
        self.assertEqual(result.missing_information[0].field_path, "claim.currency")
        self.assertFalse(result.missing_information[0].blocking)

    def test_conditional_party_and_policy_requirements_are_explicit(self):
        incident = Incident(
            id=uuid.uuid4(),
            incident_type="liability",
            event_date=datetime(2026, 8, 12, tzinfo=timezone.utc),
            location="Zurich",
        )
        result = self.service.validate(
            FactValidationRequest(
                claim=Claim(id=uuid.uuid4(), incidents=[incident]),
                claimant_reference_required=True,
                policy_reference_required=True,
            )
        )

        paths = {item.field_path for item in result.missing_information}
        self.assertIn("claim.claimant_reference", paths)
        self.assertIn("claim.policy", paths)
        self.assertFalse(result.ready_for_assessment)


if __name__ == "__main__":
    unittest.main()
