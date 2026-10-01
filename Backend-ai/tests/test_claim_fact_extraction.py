import json
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.claim_fact_extraction import ClaimFactExtractionService
from api.claim_fact_validation import ClaimFactValidationService
from api.claim_schemas import ClaimFactExtractionRequest, FactValidationRequest


class ClaimFactExtractionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def client(payload):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
        client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
            )
        )
        return client

    async def test_extracts_only_quoted_facts_and_keeps_unknowns_null(self):
        text = "On 12 August 2026 my bicycle was stolen outside Zurich station."
        service = ClaimFactExtractionService(
            client=self.client(
                {
                    "facts": [
                        {
                            "fact_path": "incidents[0].incident_type",
                            "value": "theft",
                            "confidence": 0.97,
                            "evidence": [{"source_id": "[S1]", "text_quote": "my bicycle was stolen"}],
                        },
                        {
                            "fact_path": "incidents[0].event_date",
                            "value": "2026-08-12T00:00:00",
                            "confidence": 0.94,
                            "evidence": [{"source_id": "[S1]", "text_quote": "12 August 2026"}],
                        },
                    ]
                }
            )
        )

        result = await service.extract(ClaimFactExtractionRequest(fnol_text=text))

        self.assertEqual(result.claim.incidents[0].incident_type, "theft")
        self.assertEqual(result.claim.incidents[0].event_date.year, 2026)
        self.assertEqual(result.claim.incidents[0].location, "outside Zurich station")
        self.assertEqual(len(result.claim_evidence), 3)
        claim_number = next(f for f in result.facts if f.fact_path == "claim.claim_number")
        self.assertIsNone(claim_number.value)
        self.assertEqual(claim_number.status, "unknown")
        self.assertIsNone(claim_number.confidence)

    async def test_discards_fact_when_quote_is_not_in_source(self):
        service = ClaimFactExtractionService(
            client=self.client(
                {
                    "facts": [
                        {
                            "fact_path": "incidents[0].cause",
                            "value": "negligence",
                            "confidence": 0.8,
                            "evidence": [{"source_id": "[S1]", "text_quote": "caused by negligence"}],
                        }
                    ]
                }
            )
        )

        result = await service.extract(
            ClaimFactExtractionRequest(fnol_text="Water entered the kitchen overnight.")
        )

        cause = next(f for f in result.facts if f.fact_path == "incidents[0].cause")
        self.assertEqual(cause.status, "unknown")
        self.assertIsNone(cause.value)
        self.assertEqual(result.claim_evidence, [])

    async def test_real_bicycle_fnol_accepts_groq_alias_shape(self):
        text = (
            "My bicycle was stolen outside Zurich station on 12 August 2026. "
            "The bicycle was locked. I am claiming CHF 1800 for the stolen bicycle."
        )
        service = ClaimFactExtractionService(client=self.client({
            "facts": [
                {
                    "fact_path": "claim.claimed_amount",
                    "normalized_value": "1800",
                    "confidence": 1.0,
                    "evidence": [{
                        "source_id": "S1",
                        "quote": "I am claiming CHF 1800 for the stolen bicycle.",
                    }],
                },
                {
                    "fact_path": "claim.currency",
                    "normalized_value": "CHF",
                    "confidence": 1.0,
                    "evidence": [{
                        "source_id": "S1",
                        "quote": "I am claiming CHF 1800 for the stolen bicycle.",
                    }],
                },
                {
                    "fact_path": "incidents[0].event_date",
                    "normalized_value": "2026-08-12",
                    "confidence": 1.0,
                    "evidence": [{
                        "source_id": "S1",
                        "quote": "My bicycle was stolen outside Zurich station on 12 August 2026.",
                    }],
                },
                {
                    "fact_path": "incidents[0].location",
                    "normalized_value": "Zurich station",
                    "confidence": 1.0,
                    "evidence": [{
                        "source_id": "S1",
                        "quote": "My bicycle was stolen outside Zurich station on 12 August 2026.",
                    }],
                },
                {
                    "fact_path": "incidents[0].incident_type",
                    "normalized_value": "stolen",
                    "confidence": 1.0,
                    "evidence": [{
                        "source_id": "S1",
                        "quote": "My bicycle was stolen outside Zurich station on 12 August 2026.",
                    }],
                },
            ]
        }))

        result = await service.extract(ClaimFactExtractionRequest(fnol_text=text))
        facts = {fact.fact_path: fact for fact in result.facts}

        self.assertEqual(facts["claim.claimed_amount"].value, Decimal("1800"))
        self.assertEqual(facts["claim.currency"].value, "CHF")
        self.assertEqual(result.claim.incidents[0].event_date.isoformat(), "2026-08-12T00:00:00")
        self.assertEqual(result.claim.incidents[0].location, "Zurich station")
        self.assertEqual(result.claim.incidents[0].incident_type, "stolen")
        self.assertEqual(facts["incidents[0].cause"].status, "unknown")
        self.assertEqual(result.claim_evidence[0].metadata["source_id"], "[S1]")

    async def test_e2e_bicycle_fnol_falls_back_when_groq_omits_obvious_fields(self):
        text = (
            "My bicycle was stolen outside Zurich station on 12 August 2026. "
            "The bicycle was locked. I am claiming CHF 1800 for the stolen bicycle."
        )
        service = ClaimFactExtractionService(client=self.client({
            "facts": [{
                "fact_path": "incidents[0].incident_type",
                "value": "stolen bicycle",
                "confidence": 1.0,
                "evidence": [{"source_id": "[S1]", "text_quote": "My bicycle was stolen"}],
            }]
        }))

        result = await service.extract(ClaimFactExtractionRequest(fnol_text=text))
        facts = {fact.fact_path: fact for fact in result.facts}
        validation = ClaimFactValidationService().validate(FactValidationRequest(
            claim=result.claim, facts=result.facts,
        ))

        self.assertEqual(facts["incidents[0].event_date"].value.isoformat(), "2026-08-12T00:00:00")
        self.assertEqual(facts["incidents[0].location"].value, "outside Zurich station")
        self.assertEqual(facts["claim.claimed_amount"].value, Decimal("1800"))
        self.assertEqual(facts["claim.currency"].value, "CHF")
        self.assertTrue(validation.ready_for_assessment)


if __name__ == "__main__":
    unittest.main()
