import json
import unittest
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import ValidationError

from api.claim_calculation import (
    ClaimCalculationService,
    _FinancialTerms,
    _normalize_financial_terms_payload,
)
from api.claim_schemas import Claim, ClaimCalculationRequest, ClaimFact, CoverageAssessment, EvidenceRef, Incident
from api.schemas import SemanticSearchResult


class ClaimCalculationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def client(payload):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
        client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        ))
        return client

    @staticmethod
    def policy_result(document_id):
        return SemanticSearchResult(
            chunk_id=uuid.uuid4(), document_id=document_id,
            text="Limit CHF 5000; sublimit CHF 1500; excess CHF 200.", score=0.9,
            chunk_type="text", source_file="policy.pdf", source_format="pdf",
            page_numbers=[12], section_path=["Financial terms"], sheet=None,
            row_start=None, row_end=None, email_metadata={}, rank=1,
        )

    @staticmethod
    def coverage(document_id, status="covered"):
        return CoverageAssessment(
            id=uuid.uuid4(), coverage_reference="Contents", status=status,
            policy_evidence=[EvidenceRef(
                id="[P0]", evidence_type="policy", document_id=document_id,
                source_file="policy.pdf", text_quote="Contents coverage.",
            )],
        )

    @staticmethod
    def fact(path, value, evidence_id):
        return ClaimFact(
            id=uuid.uuid4(), fact_path=path, value=value, status="confirmed",
            claim_evidence=[EvidenceRef(
                id=evidence_id, evidence_type="claim", source_file="fnol.txt",
                text_quote=f"{path}: {value}",
            )],
        )

    async def calculate(self, payload, facts, *, coverage_status="covered"):
        document_id = uuid.uuid4()
        coverage = self.coverage(document_id, coverage_status)
        retrieval = SimpleNamespace(search=AsyncMock(return_value=[self.policy_result(document_id)]))
        service = ClaimCalculationService(
            AsyncMock(), None, retrieval_service=retrieval, llm_client=self.client(payload)
        )
        response = await service.calculate(ClaimCalculationRequest(
            knowledge_base_id=uuid.uuid4(),
            claim=Claim(id=uuid.uuid4(), incidents=[Incident(id=uuid.uuid4())]),
            facts=facts, coverage_assessments=[coverage],
        ))
        return response, retrieval, coverage

    async def test_decimal_calculation_is_deterministic(self):
        facts = [
            self.fact("claim.claimed_amount", Decimal("2000"), "[C1]"),
            self.fact("claim.eligible_amount", Decimal("1800"), "[C2]"),
            self.fact("claim.currency", "CHF", "[C3]"),
            self.fact("claim.other_insurance_amount", Decimal("100"), "[C4]"),
        ]
        payload = {
            "terms_complete": True,
            "completeness_policy_evidence_ids": ["P1"],
            "deductible": {"amount": "200", "currency": "CHF", "policy_evidence_ids": ["P1"]},
            "limit": {"amount": "5000", "currency": "CHF", "policy_evidence_ids": ["P1"]},
            "sublimit": {"amount": "1500", "currency": "CHF", "policy_evidence_ids": ["P1"]},
            "percentage_limit": {"percentage": "80", "base_fact_path": "claim.claimed_amount",
                                 "policy_evidence_ids": ["P1"]},
            "other_insurance": {"rule": "subtract_known_amount", "amount": "100", "currency": "CHF",
                                "amount_fact_path": "claim.other_insurance_amount",
                                "policy_evidence_ids": ["P1"]},
        }
        response, retrieval, coverage = await self.calculate(payload, facts)
        result = response.calculation_results[0]

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.payable_amount, Decimal("1200"))
        self.assertEqual(result.claimed_amount, Decimal("2000"))
        self.assertEqual(result.eligible_amount, Decimal("1800"))
        self.assertEqual(len(result.calculation_steps), 6)
        self.assertIn("Subtract deductible", result.calculation_steps[-2])
        self.assertEqual(retrieval.search.await_args.args[1].document_id,
                         coverage.policy_evidence[0].document_id)

    async def test_missing_eligible_amount_is_incomplete(self):
        facts = [self.fact("claim.claimed_amount", Decimal("1000"), "[C1]"),
                 self.fact("claim.currency", "CHF", "[C2]")]
        response, _, _ = await self.calculate({
            "terms_complete": True, "completeness_policy_evidence_ids": ["P1"]
        }, facts)
        result = response.calculation_results[0]

        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.payable_amount)
        self.assertIn("claim.eligible_amount", [m.field_path for m in response.missing_information])

    async def test_currency_mismatch_is_incomplete(self):
        facts = [self.fact("claim.claimed_amount", Decimal("1000"), "[C1]"),
                 self.fact("claim.eligible_amount", Decimal("900"), "[C2]"),
                 self.fact("claim.currency", "CHF", "[C3]")]
        response, _, _ = await self.calculate({
            "terms_complete": True, "completeness_policy_evidence_ids": ["P1"],
            "deductible": {"amount": "100", "currency": "EUR", "policy_evidence_ids": ["P1"]},
        }, facts)

        self.assertEqual(response.calculation_results[0].status, "incomplete")
        self.assertIsNone(response.calculation_results[0].payable_amount)

    async def test_captured_scalar_deductible_is_normalized_but_not_treated_as_complete(self):
        facts = [
            self.fact("claim.claimed_amount", Decimal("1800"), "[C1]"),
            self.fact("claim.eligible_amount", Decimal("1800"), "[C2]"),
            self.fact("claim.currency", "CHF", "[C3]"),
        ]
        response, _, _ = await self.calculate({
            "deductible": "200",
            "percentage_limit": {
                "percentage": "1.0",
                "base_fact_path": "incidents[0].age",
            },
            "terms_complete": True,
            "evidence_ids": ["P8", "P10", "P11"],
        }, facts)

        result = response.calculation_results[0]
        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.payable_amount)
        self.assertIsNone(result.deductible)
        self.assertIn("policy.deductible", [
            item.field_path for item in response.missing_information
        ])

    async def test_captured_grouped_prerequisites_are_normalized_but_require_evidence(self):
        facts = [
            self.fact("incidents[0].incident_type", "theft", "[C1]"),
            self.fact("claim.claimed_amount", Decimal("1800"), "[C2]"),
            self.fact("claim.eligible_amount", Decimal("1800"), "[C3]"),
            self.fact("claim.currency", "CHF", "[C4]"),
        ]
        response, _, _ = await self.calculate({
            "terms_complete": True,
            "evidence_ids": ["P8", "P10", "P11"],
            "deductible": {
                "amount": "200",
                "currency": "CHF",
                "policy_evidence_ids": ["P11"],
            },
            "prerequisites": {
                "matched": ["incidents[0].incident_type"],
                "unmatched": [],
                "unknown": [],
            },
        }, facts)

        result = response.calculation_results[0]
        self.assertEqual(result.status, "incomplete")
        self.assertIsNone(result.payable_amount)
        self.assertTrue(any(
            item.field_path.startswith("calculation.prerequisite")
            for item in response.missing_information
        ))

    def test_captured_completeness_evidence_list_is_moved_to_its_field(self):
        raw = {
            "terms_complete": ["P8", "P10", "P12"],
            "deductible": {
                "amount": "200",
                "currency": "CHF",
                "policy_evidence_ids": ["P12"],
            },
            "prerequisites": [{
                "description": "incident_type is theft",
                "result": "matched",
                "fact_paths": ["incidents[0].incident_type"],
                "policy_evidence_ids": ["P8"],
            }],
        }
        terms = _FinancialTerms.model_validate(_normalize_financial_terms_payload(raw))

        self.assertTrue(terms.terms_complete)
        self.assertEqual(
            terms.completeness_policy_evidence_ids,
            ["P8", "P10", "P12"],
        )

    def test_canonical_financial_terms_shape_is_unchanged(self):
        raw = {
            "terms_complete": True,
            "completeness_policy_evidence_ids": ["P1"],
            "deductible": {
                "amount": "200",
                "currency": "CHF",
                "policy_evidence_ids": ["P1"],
            },
            "prerequisites": [{
                "description": "The incident is theft.",
                "result": "matched",
                "fact_paths": ["incidents[0].incident_type"],
                "policy_evidence_ids": ["P1"],
            }],
        }

        normalized = _normalize_financial_terms_payload(raw)
        terms = _FinancialTerms.model_validate(normalized)

        self.assertEqual(normalized, raw)
        self.assertEqual(terms.deductible.amount, Decimal("200"))

    def test_unsupported_financial_term_shapes_still_fail_validation(self):
        malformed = [
            {"deductible": "not-a-number"},
            {"prerequisites": {"matched": "incidents[0].incident_type"}},
            {"terms_complete": {"P1": True}},
            {
                "terms_complete": ["P1"],
                "completeness_policy_evidence_ids": ["P2"],
            },
        ]

        for raw in malformed:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                _FinancialTerms.model_validate(_normalize_financial_terms_payload(raw))

    async def test_not_covered_is_not_applicable_without_retrieval(self):
        response, retrieval, _ = await self.calculate({}, [], coverage_status="not_covered")

        self.assertEqual(response.calculation_results[0].status, "not_applicable")
        self.assertIsNone(response.calculation_results[0].payable_amount)
        retrieval.search.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
