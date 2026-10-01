"""Phase 3 applicable-policy and basic coverage-trigger assessment."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from groq import APIError
from pydantic import BaseModel, Field, model_validator

from api.claim_schemas import (
    ApplicablePolicyAssessment,
    ClaimFact,
    CoverageAssessment,
    CoverageAssessmentRequest,
    CoverageAssessmentResponse,
    EvidenceRef,
    MissingInformation,
)
from api.llm_service import GroundedAnswerService, LLMProviderError
from api.retrieval import SemanticRetrievalService
from api.schemas import SemanticSearchRequest, SemanticSearchResult


class _TriggerCondition(BaseModel):
    description: str
    result: str
    fact_paths: list[str] = Field(default_factory=list)
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _CoverageCandidate(BaseModel):
    coverage_reference: str
    conditions: list[_TriggerCondition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_flat_condition_shape(cls, value):
        if not isinstance(value, dict) or "coverage_reference" in value:
            return value
        if "description" not in value:
            return value
        description = value["description"]
        return {
            "coverage_reference": description,
            "conditions": [{
                "description": description,
                "result": value.get("result", "unknown"),
                "fact_paths": value.get("fact_paths", []),
                "policy_evidence_ids": value.get("policy_evidence_ids", []),
            }],
        }


class _CoveragePayload(BaseModel):
    candidates: list[_CoverageCandidate] = Field(default_factory=list)


class CoverageAssessmentService:
    def __init__(
        self,
        session,
        embedding_model,
        *,
        retrieval_service=None,
        llm_client=None,
        llm_model: str | None = None,
    ):
        self.retrieval = retrieval_service or SemanticRetrievalService(session, embedding_model)
        configured = GroundedAnswerService(client=llm_client, model=llm_model)
        self.client = configured.client
        self.model = configured.model

    async def assess(self, request: CoverageAssessmentRequest) -> CoverageAssessmentResponse:
        fact_map = {
            fact.fact_path: fact
            for fact in request.facts
            if fact.value is not None and fact.status != "unknown"
        }
        policy_results = await self.retrieval.search(
            request.knowledge_base_id,
            SemanticSearchRequest(
                query=self._policy_query(request, fact_map),
                limit=request.retrieval_limit,
                retrieval_mode="hybrid_rerank",
            ),
        )
        missing: list[MissingInformation] = []
        policy_assessment, policy_missing = self._select_policy(request, policy_results)
        if policy_missing:
            missing.append(policy_missing)
            policy_assessment.missing_information_ids.append(policy_missing.id)

        coverage_results = await self.retrieval.search(
            request.knowledge_base_id,
            SemanticSearchRequest(
                query=self._coverage_query(request, fact_map),
                limit=request.retrieval_limit,
                retrieval_mode="hybrid_rerank",
                document_id=policy_assessment.document_id,
            ),
        )
        policy_evidence, evidence_by_id = self._policy_evidence(coverage_results)
        payload = await self._extract_candidates(request, fact_map, policy_evidence)
        assessments = self._build_assessments(
            payload,
            policy_assessment.status == "identified",
            fact_map,
            evidence_by_id,
            missing,
        )
        if not assessments:
            item = self._missing(
                field_path="coverage.candidate",
                reason="No affirmative coverage section could be identified from the retrieved policy evidence.",
                required_for="coverage_assessment",
                question="Which policy coverage is this claim being made under?",
            )
            missing.append(item)

        return CoverageAssessmentResponse(
            claim_id=request.claim.id,
            applicable_policy=policy_assessment,
            candidate_policy_sources=list(dict.fromkeys(r.source_file for r in policy_results)),
            coverage_assessments=assessments,
            missing_information=missing,
        )

    @staticmethod
    def _policy_query(request: CoverageAssessmentRequest, facts: dict[str, ClaimFact]) -> str:
        identifiers = CoverageAssessmentService._policy_identifiers(request)
        fact_text = CoverageAssessmentService._fact_text(facts)
        return (
            "Identify the policy document and version applicable to this claim. "
            f"Explicit policy identifiers: {', '.join(identifiers) or 'none supplied'}. "
            f"Explicit claim facts: {fact_text or 'none supplied'}."
        )

    @staticmethod
    def _coverage_query(request: CoverageAssessmentRequest, facts: dict[str, ClaimFact]) -> str:
        incident_text = "; ".join(
            f"incident_type={incident.incident_type}, event_date={incident.event_date}, "
            f"location={incident.location}, cause={incident.cause}"
            for incident in request.claim.incidents
        )
        return (
            "Find affirmative policy coverage sections and their basic trigger conditions for this claim. "
            "Do not evaluate exclusions, duties, deductibles, limits, or payment. "
            f"Incidents: {incident_text or 'none supplied'}. Facts: "
            f"{CoverageAssessmentService._fact_text(facts) or 'none supplied'}."
        )

    @staticmethod
    def _fact_text(facts: dict[str, ClaimFact]) -> str:
        return "; ".join(f"{path}={fact.value}" for path, fact in facts.items())

    @staticmethod
    def _policy_identifiers(request: CoverageAssessmentRequest) -> list[str]:
        policy = request.claim.policy
        if policy is None:
            return []
        return [
            value
            for value in (policy.policy_number, policy.policy_version, policy.product_code)
            if value
        ]

    @classmethod
    def _select_policy(
        cls,
        request: CoverageAssessmentRequest,
        results: list[SemanticSearchResult],
    ) -> tuple[ApplicablePolicyAssessment, MissingInformation | None]:
        identifiers = cls._policy_identifiers(request)
        for result in results:
            searchable = f"{result.source_file}\n{result.text}".casefold()
            if identifiers and all(identifier.casefold() in searchable for identifier in identifiers):
                evidence = cls._evidence_from_result("[P1]", result)
                return (
                    ApplicablePolicyAssessment(
                        id=uuid.uuid4(),
                        status="identified",
                        policy=request.claim.policy,
                        document_id=result.document_id,
                        source_file=result.source_file,
                        policy_evidence=[evidence],
                    ),
                    None,
                )

        status = "candidate" if results else "indeterminate"
        missing = cls._missing(
            field_path="claim.policy",
            reason="The supplied claim information does not uniquely identify an applicable policy document and version.",
            required_for="policy_applicability",
            question="What policy number and policy version apply to this claim?",
        )
        return ApplicablePolicyAssessment(id=uuid.uuid4(), status=status), missing

    async def _extract_candidates(
        self,
        request: CoverageAssessmentRequest,
        facts: dict[str, ClaimFact],
        evidence: list[EvidenceRef],
    ) -> _CoveragePayload:
        evidence_text = "\n\n".join(
            f"{item.id}\nSource: {item.source_file}\nSection: {' > '.join(item.section_path)}\n{item.text_quote}"
            for item in evidence
        )
        fact_text = "\n".join(f"{path}: {fact.value}" for path, fact in facts.items())
        prompt = (
            "Return candidate affirmative coverages only in this nested JSON shape: "
            "{\"candidates\":[{\"coverage_reference\":\"...\",\"conditions\":[{"
            "\"description\":\"...\",\"result\":\"matched|unmatched|unknown\","
            "\"fact_paths\":[],\"policy_evidence_ids\":[]}]}]}. Keep coverage_reference at "
            "candidate level and trigger fields inside conditions. Do not consider exclusions, obligations, deductibles, limits, "
            "other insurance, or payment. Use only supplied facts and evidence.\n\n"
            f"Claim facts:\n{fact_text or 'none'}\n\nPolicy evidence:\n{evidence_text or 'none'}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "Extract basic coverage triggers conservatively. Return JSON with a candidates array.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except APIError as error:
            raise LLMProviderError(f"Groq request failed: {error}") from error
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty coverage assessment")
        return _CoveragePayload.model_validate(json.loads(content))

    @classmethod
    def _build_assessments(
        cls,
        payload: _CoveragePayload,
        policy_identified: bool,
        facts: dict[str, ClaimFact],
        policy_evidence: dict[str, EvidenceRef],
        missing: list[MissingInformation],
    ) -> list[CoverageAssessment]:
        assessments = []
        for candidate in payload.candidates:
            matched, unmatched, unknown = [], [], []
            used_policy: dict[str, EvidenceRef] = {}
            used_claim: dict[str, EvidenceRef] = {}
            missing_ids = []
            for index, condition in enumerate(candidate.conditions):
                evidence_ids = [cls._bracket(item) for item in condition.policy_evidence_ids]
                valid_policy = [policy_evidence[item] for item in evidence_ids if item in policy_evidence]
                if not valid_policy:
                    continue
                for item in valid_policy:
                    used_policy[item.id] = item

                missing_paths = [path for path in condition.fact_paths if path not in facts]
                result = condition.result.casefold()
                if missing_paths or result not in {"matched", "unmatched", "unknown"}:
                    result = "unknown"
                if result in {"matched", "unmatched"} and not condition.fact_paths:
                    result = "unknown"

                if result == "matched":
                    matched.append(condition.description)
                elif result == "unmatched":
                    unmatched.append(condition.description)
                else:
                    unknown.append(condition.description)
                    field_path = missing_paths[0] if missing_paths else (
                        condition.fact_paths[0] if condition.fact_paths else
                        f"coverage.{cls._slug(candidate.coverage_reference)}.condition_{index + 1}"
                    )
                    item = cls._missing(
                        field_path=field_path,
                        reason=f"Information is needed to evaluate coverage condition: {condition.description}",
                        required_for="coverage_assessment",
                        question=f"Please provide information for: {condition.description}",
                    )
                    missing.append(item)
                    missing_ids.append(item.id)

                for path in condition.fact_paths:
                    fact = facts.get(path)
                    if fact:
                        for item in fact.claim_evidence:
                            used_claim[item.id] = item

            if unmatched:
                status = "not_covered"
            elif matched and not unknown and policy_identified:
                status = "covered"
            elif matched and not unmatched:
                status = "potentially_covered"
            else:
                status = "indeterminate"

            assessments.append(
                CoverageAssessment(
                    id=uuid.uuid4(),
                    coverage_reference=candidate.coverage_reference,
                    status=status,
                    matched_conditions=matched,
                    unmatched_conditions=unmatched,
                    unknown_conditions=unknown,
                    policy_evidence=list(used_policy.values()),
                    claim_evidence=list(used_claim.values()),
                    missing_information_ids=missing_ids,
                )
            )
        return assessments

    @classmethod
    def _policy_evidence(
        cls, results: list[SemanticSearchResult]
    ) -> tuple[list[EvidenceRef], dict[str, EvidenceRef]]:
        evidence = []
        seen = set()
        for result in results:
            items: list[Any] = [result, *result.context_chunks]
            for item in items:
                if item.chunk_id in seen:
                    continue
                seen.add(item.chunk_id)
                evidence.append(cls._evidence_from_result(f"[P{len(evidence) + 1}]", item))
        return evidence, {item.id: item for item in evidence}

    @staticmethod
    def _evidence_from_result(evidence_id: str, result) -> EvidenceRef:
        return EvidenceRef(
            id=evidence_id,
            evidence_type="policy",
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            source_file=result.source_file,
            page_numbers=result.page_numbers,
            section_path=result.section_path,
            text_quote=result.text,
        )

    @staticmethod
    def _missing(*, field_path: str, reason: str, required_for: str, question: str) -> MissingInformation:
        return MissingInformation(
            id=uuid.uuid4(),
            field_path=field_path,
            reason=reason,
            required_for=required_for,
            blocking=True,
            question=question,
        )

    @staticmethod
    def _bracket(value: str) -> str:
        return f"[{value}]" if re.fullmatch(r"P\d+", value) else value

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "candidate"
