import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.claim_fact_extraction import ClaimFactExtractionService
from api.claim_schemas import ClaimFactExtractionRequest


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
        self.assertEqual(len(result.claim_evidence), 2)
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


if __name__ == "__main__":
    unittest.main()
