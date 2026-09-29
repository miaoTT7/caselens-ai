"""Phase 6 deterministic, evidence-grounded claim calculation."""

from __future__ import annotations

import json
import re
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from groq import APIError
from pydantic import BaseModel, Field

from api.claim_schemas import (
    CalculationResult,
    ClaimCalculationRequest,
    ClaimCalculationResponse,
    ClaimFact,
    CoverageAssessment,
    EvidenceRef,
    MissingInformation,
)
from api.llm_service import GroundedAnswerService, LLMProviderError
from api.retrieval import SemanticRetrievalService
from api.schemas import SemanticSearchRequest, SemanticSearchResult


class _MoneyTerm(BaseModel):
    amount: Decimal | None = None
    currency: str | None = None
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _PercentageTerm(BaseModel):
    percentage: Decimal | None = None
    base_fact_path: str | None = None
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _OtherInsuranceTerm(BaseModel):
    rule: str
    amount: Decimal | None = None
    amount_fact_path: str | None = None
    currency: str | None = None
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _Prerequisite(BaseModel):
    description: str
    result: str
    fact_paths: list[str] = Field(default_factory=list)
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _FinancialTerms(BaseModel):
    terms_complete: bool = False
    completeness_policy_evidence_ids: list[str] = Field(default_factory=list)
    deductible: _MoneyTerm | None = None
    limit: _MoneyTerm | None = None
    sublimit: _MoneyTerm | None = None
    percentage_limit: _PercentageTerm | None = None
    other_insurance: _OtherInsuranceTerm | None = None
    prerequisites: list[_Prerequisite] = Field(default_factory=list)


class ClaimCalculationService:
    def __init__(self, session, embedding_model, *, retrieval_service=None, llm_client=None,
                 llm_model: str | None = None):
        self.retrieval = retrieval_service or SemanticRetrievalService(session, embedding_model)
        configured = GroundedAnswerService(client=llm_client, model=llm_model)
        self.client = configured.client
        self.model = configured.model

    async def calculate(self, request: ClaimCalculationRequest) -> ClaimCalculationResponse:
        facts = {f.fact_path: f for f in request.facts if f.value is not None and f.status != "unknown"}
        results: list[CalculationResult] = []
        missing: list[MissingInformation] = []
        active_coverages = [c for c in request.coverage_assessments if c.status != "not_covered"]

        for coverage in request.coverage_assessments:
            if coverage.status == "not_covered":
                results.append(self._not_applicable(coverage))
                continue
            document_id = self._document_id(coverage)
            if document_id is None:
                results.append(self._incomplete(request, coverage, facts, missing,
                    "claim.policy.document_id", "A unique applicable policy document is required."))
                continue
            retrieved = await self.retrieval.search(
                request.knowledge_base_id,
                SemanticSearchRequest(
                    query=self._query(request, coverage, facts), limit=request.retrieval_limit,
                    retrieval_mode="hybrid_rerank", document_id=document_id,
                ),
            )
            evidence, evidence_map = self._policy_evidence(retrieved)
            terms = await self._extract_terms(request, coverage, facts, evidence)
            results.append(self._calculate_one(
                request, coverage, facts, terms, evidence_map, missing, len(active_coverages)
            ))
        return ClaimCalculationResponse(
            claim_id=request.claim.id, calculation_results=results, missing_information=missing
        )

    @staticmethod
    def _document_id(coverage: CoverageAssessment) -> uuid.UUID | None:
        ids = {e.document_id for e in coverage.policy_evidence if e.document_id is not None}
        return next(iter(ids)) if len(ids) == 1 else None

    @staticmethod
    def _query(request, coverage, facts) -> str:
        exclusions = "; ".join(
            f"{e.exclusion_reference}={e.status}" for e in request.exclusion_assessments
            if e.coverage_assessment_id == coverage.id
        )
        obligations = "; ".join(
            f"{o.obligation_reference}={o.status}; consequence={o.permitted_consequence}"
            for o in request.obligation_assessments if o.coverage_assessment_id == coverage.id
        )
        fact_text = "; ".join(f"{path}={fact.value}" for path, fact in facts.items())
        return (
            f"Find all financial terms needed to calculate coverage '{coverage.coverage_reference}': "
            "deductible or excess, limit, sublimit, percentage cap, other-insurance rule, and calculation "
            f"prerequisites. Claim facts: {fact_text or 'none'}. Exclusions: {exclusions or 'none'}. "
            f"Obligations: {obligations or 'none'}. Do not calculate a payable amount."
        )

    async def _extract_terms(self, request, coverage, facts, evidence) -> _FinancialTerms:
        evidence_text = "\n\n".join(
            f"{e.id}\nSource: {e.source_file}\nSection: {' > '.join(e.section_path)}\n{e.text_quote}"
            for e in evidence
        )
        fact_text = "\n".join(f"{path}: {fact.value}" for path, fact in facts.items())
        prompt = (
            f"Coverage: {coverage.coverage_reference}\nExtract calculation inputs only. Monetary values and "
            "percentages must be decimal strings. Return terms_complete and evidence IDs showing that the "
            "retrieved clauses contain the complete relevant calculation basis. Optional fields are deductible, "
            "limit, sublimit, percentage_limit (percentage and base_fact_path), other_insurance (rule must be "
            "subtract_known_amount, no_reduction, coordination_required, pro_rata, or unknown), and prerequisites "
            "For an other-insurance amount, also return the exact amount_fact_path from Claim facts. "
            "with matched/unmatched/unknown result and fact paths. Omit a term when it is not stated. Never "
            "calculate payable amount or infer a missing amount/rule. Return one JSON object.\n\n"
            f"Claim facts:\n{fact_text or 'none'}\n\nPolicy evidence:\n{evidence_text or 'none'}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model, temperature=0, response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Extract grounded insurance calculation inputs only."},
                    {"role": "user", "content": prompt},
                ],
            )
        except APIError as error:
            raise LLMProviderError(f"Groq request failed: {error}") from error
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned empty calculation terms")
        return _FinancialTerms.model_validate(json.loads(content))

    @classmethod
    def _calculate_one(cls, request, coverage, facts, terms, evidence_map, missing, active_count):
        missing_ids: list[uuid.UUID] = []
        used_policy: dict[str, EvidenceRef] = {}
        used_claim: dict[str, EvidenceRef] = {}
        steps: list[str] = []
        claimed_fact = cls._amount_fact(facts, coverage, "claimed_amount", active_count)
        eligible_fact = cls._amount_fact(facts, coverage, "eligible_amount", active_count)
        currency_fact = cls._value_fact(facts, coverage, "currency", active_count)

        exposure = next((e for e in request.claim.exposures if e.id == coverage.exposure_id), None)
        claimed = cls._decimal(claimed_fact.value) if claimed_fact else (
            exposure.claimed_amount if exposure else None
        )
        eligible = cls._decimal(eligible_fact.value) if eligible_fact else None
        currency = str(currency_fact.value).upper() if currency_fact else (
            exposure.currency.upper() if exposure and exposure.currency else None
        )
        for fact in (claimed_fact, eligible_fact, currency_fact):
            if fact:
                cls._collect_claim(fact, used_claim)

        for path, value, reason in (
            ("claim.claimed_amount", claimed, "The claimed amount is required."),
            ("claim.eligible_amount", eligible, "The eligible covered-loss amount is required."),
            ("claim.currency", currency, "The claim currency is required."),
        ):
            if value is None:
                cls._add_missing(request, coverage, missing, missing_ids, path, reason)
        if claimed is not None and claimed < 0:
            cls._add_missing(request, coverage, missing, missing_ids,
                             "claim.claimed_amount", "Claimed amount cannot be negative.")
        if eligible is not None and eligible < 0:
            cls._add_missing(request, coverage, missing, missing_ids,
                             "claim.eligible_amount", "Eligible amount cannot be negative.")
        if claimed is not None and eligible is not None and eligible > claimed:
            cls._add_missing(request, coverage, missing, missing_ids,
                             "claim.eligible_amount",
                             "Eligible amount exceeds the claimed amount and must be reconciled.")

        completeness = cls._valid_evidence(terms.completeness_policy_evidence_ids, evidence_map)
        if not terms.terms_complete or not completeness:
            cls._add_missing(request, coverage, missing, missing_ids,
                             "policy.financial_terms", "The complete calculation basis is not supported by policy evidence.")
        cls._collect_policy(completeness, used_policy)

        for prerequisite in terms.prerequisites:
            policy = cls._valid_evidence(prerequisite.policy_evidence_ids, evidence_map)
            fact_paths_valid = prerequisite.fact_paths and all(p in facts for p in prerequisite.fact_paths)
            result = prerequisite.result.casefold()
            if not policy or not fact_paths_valid or result != "matched":
                path = next((p for p in prerequisite.fact_paths if p not in facts),
                            f"calculation.prerequisite.{cls._slug(prerequisite.description)}")
                cls._add_missing(request, coverage, missing, missing_ids, path,
                                 f"Calculation prerequisite is not established: {prerequisite.description}")
            cls._collect_policy(policy, used_policy)
            for path in prerequisite.fact_paths:
                if path in facts:
                    cls._collect_claim(facts[path], used_claim)

        current = eligible
        policy_currency = currency
        money_terms = (("limit", terms.limit), ("sublimit", terms.sublimit))
        validated: dict[str, Decimal] = {}
        deductible = None
        other_amount = None

        for name, term in (("deductible", terms.deductible), *money_terms):
            if term is None:
                continue
            amount, policy = cls._validate_money_term(term, currency, evidence_map)
            cls._collect_policy(policy, used_policy)
            if amount is None:
                cls._add_missing(request, coverage, missing, missing_ids, f"policy.{name}",
                                 f"The {name} amount, currency, or policy evidence is invalid or missing.")
            else:
                validated[name] = amount
                policy_currency = term.currency.upper()

        percentage_cap = None
        if terms.percentage_limit is not None:
            term = terms.percentage_limit
            policy = cls._valid_evidence(term.policy_evidence_ids, evidence_map)
            base = facts.get(term.base_fact_path or "")
            percent = cls._decimal(term.percentage)
            base_amount = cls._decimal(base.value) if base else None
            if not policy or percent is None or percent < 0 or base_amount is None or base_amount < 0:
                cls._add_missing(request, coverage, missing, missing_ids,
                                 term.base_fact_path or "policy.percentage_limit",
                                 "The percentage limit or its explicit base is missing or invalid.")
            else:
                percentage_cap = base_amount * percent / Decimal("100")
                cls._collect_policy(policy, used_policy)
                cls._collect_claim(base, used_claim)

        if terms.other_insurance is not None:
            other = terms.other_insurance
            policy = cls._valid_evidence(other.policy_evidence_ids, evidence_map)
            rule = other.rule.casefold()
            cls._collect_policy(policy, used_policy)
            if not policy or rule not in {"subtract_known_amount", "no_reduction"}:
                cls._add_missing(request, coverage, missing, missing_ids, "policy.other_insurance",
                                 "The other-insurance rule is missing or requires unsupported coordination.")
            elif rule == "subtract_known_amount":
                other_fact = facts.get(other.amount_fact_path or "")
                other_amount = cls._decimal(other_fact.value) if other_fact else None
                extracted_amount = cls._decimal(other.amount)
                if (other_amount is None or extracted_amount != other_amount or other_amount < 0
                        or not other.currency or not currency
                        or other.currency.upper() != currency):
                    other_amount = None
                    cls._add_missing(request, coverage, missing, missing_ids, "claim.other_insurance_amount",
                                     "A supported other-insurance amount in the claim currency is required.")
                else:
                    cls._collect_claim(other_fact, used_claim)

        consequential_breaches = [
            obligation for obligation in request.obligation_assessments
            if obligation.coverage_assessment_id == coverage.id
            and obligation.status == "breached"
            and obligation.permitted_consequence
        ]
        if consequential_breaches:
            cls._add_missing(
                request, coverage, missing, missing_ids,
                "policy.obligation_financial_consequence",
                "A permitted consequence exists but is not represented as a supported deterministic operation.",
            )

        if missing_ids or current is None or currency is None:
            payable = None
            status = "incomplete"
        else:
            steps.append(f"Start with eligible amount: {cls._money(currency, current)}")
            if percentage_cap is not None:
                before = current
                current = min(current, percentage_cap)
                steps.append(f"Apply percentage limit: min({cls._money(currency, before)}, "
                             f"{cls._money(currency, percentage_cap)}) = {cls._money(currency, current)}")
            for name in ("limit", "sublimit"):
                if name in validated:
                    before = current
                    current = min(current, validated[name])
                    steps.append(f"Apply {name}: min({cls._money(currency, before)}, "
                                 f"{cls._money(currency, validated[name])}) = {cls._money(currency, current)}")
            deductible = validated.get("deductible")
            if deductible is not None:
                before = current
                current = max(current - deductible, Decimal("0"))
                steps.append(f"Subtract deductible: max({cls._money(currency, before)} - "
                             f"{cls._money(currency, deductible)}, 0) = {cls._money(currency, current)}")
            if other_amount is not None:
                before = current
                current = max(current - other_amount, Decimal("0"))
                steps.append(f"Subtract other insurance: max({cls._money(currency, before)} - "
                             f"{cls._money(currency, other_amount)}, 0) = {cls._money(currency, current)}")
            payable, status = current, "complete"

        return CalculationResult(
            id=uuid.uuid4(), coverage_assessment_id=coverage.id, exposure_id=coverage.exposure_id,
            status=status, currency=currency or policy_currency, claimed_amount=claimed,
            eligible_amount=eligible, deductible=validated.get("deductible"),
            limit=validated.get("limit"), sublimit=validated.get("sublimit"),
            other_insurance_amount=other_amount, payable_amount=payable,
            calculation_steps=steps, policy_evidence=list(used_policy.values()),
            claim_evidence=list(used_claim.values()), missing_information_ids=missing_ids,
        )

    @classmethod
    def _incomplete(cls, request, coverage, facts, missing, path, reason):
        ids: list[uuid.UUID] = []
        cls._add_missing(request, coverage, missing, ids, path, reason)
        return CalculationResult(id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            exposure_id=coverage.exposure_id, status="incomplete", missing_information_ids=ids)

    @staticmethod
    def _not_applicable(coverage):
        return CalculationResult(id=uuid.uuid4(), coverage_assessment_id=coverage.id,
            exposure_id=coverage.exposure_id, status="not_applicable")

    @staticmethod
    def _value_fact(facts, coverage, name, active_count):
        paths = [f"coverage_assessments.{coverage.id}.{name}"]
        if coverage.exposure_id:
            paths.append(f"exposures.{coverage.exposure_id}.{name}")
        if active_count == 1:
            paths.append(f"claim.{name}")
        return next((facts[p] for p in paths if p in facts), None)

    @classmethod
    def _amount_fact(cls, facts, coverage, name, active_count):
        return cls._value_fact(facts, coverage, name, active_count)

    @staticmethod
    def _decimal(value) -> Decimal | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            result = value if isinstance(value, Decimal) else Decimal(str(value))
            return result if result.is_finite() else None
        except (InvalidOperation, ValueError):
            return None

    @classmethod
    def _validate_money_term(cls, term, currency, evidence_map):
        amount = cls._decimal(term.amount)
        evidence = cls._valid_evidence(term.policy_evidence_ids, evidence_map)
        if (amount is None or amount < 0 or not term.currency or not currency
                or term.currency.upper() != currency or not evidence):
            return None, evidence
        return amount, evidence

    @classmethod
    def _valid_evidence(cls, ids, evidence_map):
        return [evidence_map[key] for key in (cls._bracket(v) for v in ids) if key in evidence_map]

    @staticmethod
    def _collect_policy(items, target):
        for item in items:
            target[item.id] = item

    @staticmethod
    def _collect_claim(fact, target):
        for item in fact.claim_evidence:
            target[item.id] = item

    @classmethod
    def _policy_evidence(cls, results: list[SemanticSearchResult]):
        evidence, seen = [], set()
        for result in results:
            for item in [result, *result.context_chunks]:
                if item.chunk_id in seen:
                    continue
                seen.add(item.chunk_id)
                evidence.append(EvidenceRef(id=f"[P{len(evidence)+1}]", evidence_type="policy",
                    document_id=item.document_id, chunk_id=item.chunk_id, source_file=item.source_file,
                    page_numbers=item.page_numbers, section_path=item.section_path, text_quote=item.text))
        return evidence, {e.id: e for e in evidence}

    @classmethod
    def _add_missing(cls, request, coverage, missing, ids, path, reason):
        incident_id = None
        if coverage.exposure_id:
            exposure = next((e for e in request.claim.exposures if e.id == coverage.exposure_id), None)
            incident_id = exposure.incident_id if exposure else None
        if incident_id is None and len(request.claim.incidents) == 1:
            incident_id = request.claim.incidents[0].id
        item = MissingInformation(id=uuid.uuid4(), field_path=path, reason=reason,
            required_for="claim_calculation", blocking=True, related_incident_id=incident_id,
            related_exposure_id=coverage.exposure_id)
        missing.append(item)
        ids.append(item.id)

    @staticmethod
    def _money(currency, amount):
        return f"{currency} {amount.quantize(Decimal('0.01'))}"

    @staticmethod
    def _bracket(value):
        return f"[{value}]" if re.fullmatch(r"P\d+", value) else value

    @staticmethod
    def _slug(value):
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "item"
