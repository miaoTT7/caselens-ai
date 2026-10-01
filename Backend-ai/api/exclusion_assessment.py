"""Phase 4 policy exclusion assessment for existing coverage candidates."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from groq import APIError
from pydantic import BaseModel, Field, field_validator

from api.claim_schemas import (
    ClaimFact,
    CoverageAssessment,
    EvidenceRef,
    ExclusionAssessment,
    ExclusionAssessmentRequest,
    ExclusionAssessmentResponse,
    MissingInformation,
)
from api.llm_service import GroundedAnswerService, LLMProviderError
from api.retrieval import SemanticRetrievalService
from api.schemas import SemanticSearchRequest, SemanticSearchResult


class _ExclusionCondition(BaseModel):
    description: str
    result: str
    fact_paths: list[str] = Field(default_factory=list)
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _ExclusionCandidate(BaseModel):
    exclusion_reference: str
    condition_logic: Literal["all", "any"] | str
    conditions: list[_ExclusionCondition] = Field(default_factory=list)


class _ExclusionPayload(BaseModel):
    exclusions: list[_ExclusionCandidate] = Field(default_factory=list)

    @field_validator("exclusions", mode="before")
    @classmethod
    def remove_empty_string_entries(cls, value):
        if not isinstance(value, list):
            return value
        return [item for item in value if not (isinstance(item, str) and not item.strip())]


class ExclusionAssessmentService:
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

    async def assess(self, request: ExclusionAssessmentRequest) -> ExclusionAssessmentResponse:
        fact_map = {
            fact.fact_path: fact
            for fact in request.facts
            if fact.value is not None and fact.status != "unknown"
        }
        assessments: list[ExclusionAssessment] = []
        missing: list[MissingInformation] = []

        for coverage in request.coverage_assessments:
            if coverage.status == "not_covered":
                continue
            document_id = self._coverage_document_id(coverage)
            results = await self.retrieval.search(
                request.knowledge_base_id,
                SemanticSearchRequest(
                    query=self._exclusion_query(request, coverage, fact_map),
                    limit=request.retrieval_limit,
                    retrieval_mode="hybrid_rerank",
                    document_id=document_id,
                ),
            )
            evidence, evidence_by_id = self._policy_evidence(results)
            payload = await self._extract_exclusions(coverage, fact_map, evidence)
            coverage_assessments = self._build_assessments(
                request, coverage, payload, fact_map, evidence_by_id, missing
            )
            assessments.extend(coverage_assessments)

            if not coverage_assessments:
                missing.append(
                    self._missing(
                        field_path=f"exclusions.{self._slug(coverage.coverage_reference)}",
                        reason=(
                            "No relevant exclusion clause could be established from the retrieved "
                            f"policy evidence for coverage {coverage.coverage_reference}."
                        ),
                        question=None,
                        related_incident_id=self._related_incident_id(request, coverage),
                        related_exposure_id=coverage.exposure_id,
                    )
                )

        return ExclusionAssessmentResponse(
            claim_id=request.claim.id,
            exclusion_assessments=assessments,
            missing_information=missing,
        )

    @staticmethod
    def _coverage_document_id(coverage: CoverageAssessment) -> uuid.UUID | None:
        document_ids = {
            item.document_id for item in coverage.policy_evidence if item.document_id is not None
        }
        return next(iter(document_ids)) if len(document_ids) == 1 else None

    @staticmethod
    def _exclusion_query(
        request: ExclusionAssessmentRequest,
        coverage: CoverageAssessment,
        facts: dict[str, ClaimFact],
    ) -> str:
        incidents = "; ".join(
            f"type={item.incident_type}, cause={item.cause}, location={item.location}"
            for item in request.claim.incidents
        )
        fact_text = "; ".join(f"{path}={fact.value}" for path, fact in facts.items())
        return (
            f"Find policy exclusions specifically relevant to coverage '{coverage.coverage_reference}' "
            f"and this incident. Incidents: {incidents or 'none supplied'}. "
            f"Explicit claim facts: {fact_text or 'none supplied'}. "
            "Return exclusion clauses only; ignore obligations, deductibles, limits, other insurance, "
            "payment calculations, and recommendations."
        )

    async def _extract_exclusions(
        self,
        coverage: CoverageAssessment,
        facts: dict[str, ClaimFact],
        evidence: list[EvidenceRef],
    ) -> _ExclusionPayload:
        fact_text = "\n".join(f"{path}: {fact.value}" for path, fact in facts.items())
        evidence_text = "\n\n".join(
            f"{item.id}\nSource: {item.source_file}\n"
            f"Section: {' > '.join(item.section_path)}\n{item.text_quote}"
            for item in evidence
        )
        prompt = (
            f"Coverage: {coverage.coverage_reference}\n"
            "Extract only exclusions relevant to this coverage and incident. Return this nested JSON "
            "shape: {\"exclusions\":[{\"exclusion_reference\":\"...\",\"condition_logic\":\"all|any\","
            "\"conditions\":[{\"description\":\"...\",\"result\":\"matched|unmatched|unknown\","
            "\"fact_paths\":[],\"policy_evidence_ids\":[]}]}]}. A matched or unmatched result "
            "requires explicit claim facts and explicit policy evidence. Absence of a fact is unknown. "
            "The exclusions array may contain JSON objects only; never include strings, empty entries, "
            "or placeholders. "
            "Do not evaluate obligations, deductibles, limits, other insurance, payment, or make a final "
            "claim decision. Return JSON with an exclusions array.\n\n"
            f"Claim facts:\n{fact_text or 'none'}\n\nPolicy evidence:\n{evidence_text or 'none'}"
        )
        request_args = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract policy exclusion conditions conservatively. Use only supplied facts "
                        "and evidence; return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = await self.client.chat.completions.create(**request_args)
        except APIError as first_error:
            if not self._is_json_generation_error(first_error):
                raise LLMProviderError(f"Groq request failed: {first_error}") from first_error
            try:
                response = await self.client.chat.completions.create(**request_args)
            except APIError as retry_error:
                raise LLMProviderError(f"Groq request failed: {retry_error}") from retry_error
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty exclusion assessment")
        return _ExclusionPayload.model_validate(json.loads(content))

    @staticmethod
    def _is_json_generation_error(error: APIError) -> bool:
        return getattr(error, "status_code", None) == 400 and "json_validate_failed" in str(error)

    @classmethod
    def _build_assessments(
        cls,
        request: ExclusionAssessmentRequest,
        coverage: CoverageAssessment,
        payload: _ExclusionPayload,
        facts: dict[str, ClaimFact],
        policy_evidence: dict[str, EvidenceRef],
        missing: list[MissingInformation],
    ) -> list[ExclusionAssessment]:
        assessments: list[ExclusionAssessment] = []
        for candidate in payload.exclusions:
            matched: list[str] = []
            unmatched: list[str] = []
            unknown: list[str] = []
            used_policy: dict[str, EvidenceRef] = {}
            used_claim: dict[str, EvidenceRef] = {}
            missing_ids: list[uuid.UUID] = []

            for index, condition in enumerate(candidate.conditions):
                evidence_ids = [cls._bracket(item) for item in condition.policy_evidence_ids]
                valid_policy = [policy_evidence[item] for item in evidence_ids if item in policy_evidence]
                missing_paths = [path for path in condition.fact_paths if path not in facts]
                result = condition.result.casefold()

                if (
                    not valid_policy
                    or missing_paths
                    or not condition.fact_paths
                    or result not in {"matched", "unmatched", "unknown"}
                ):
                    result = "unknown"

                for item in valid_policy:
                    used_policy[item.id] = item
                for path in condition.fact_paths:
                    fact = facts.get(path)
                    if fact:
                        for item in fact.claim_evidence:
                            used_claim[item.id] = item

                if result == "matched":
                    matched.append(condition.description)
                elif result == "unmatched":
                    unmatched.append(condition.description)
                else:
                    unknown.append(condition.description)
                    field_path = missing_paths[0] if missing_paths else (
                        condition.fact_paths[0] if condition.fact_paths else
                        f"exclusions.{cls._slug(candidate.exclusion_reference)}.condition_{index + 1}"
                    )
                    item = cls._missing(
                        field_path=field_path,
                        reason=(
                            "Information or grounded policy evidence is needed to evaluate exclusion "
                            f"condition: {condition.description}"
                        ),
                        question=f"Please provide information for: {condition.description}",
                        related_incident_id=cls._related_incident_id(request, coverage),
                        related_exposure_id=coverage.exposure_id,
                    )
                    missing.append(item)
                    missing_ids.append(item.id)

            logic = candidate.condition_logic.casefold()
            if not candidate.conditions or logic not in {"all", "any"}:
                item = cls._missing(
                    field_path=f"exclusions.{cls._slug(candidate.exclusion_reference)}.conditions",
                    reason=(
                        "The exclusion conditions or their required relationship could not be "
                        "established from grounded policy evidence."
                    ),
                    question=None,
                    related_incident_id=cls._related_incident_id(request, coverage),
                    related_exposure_id=coverage.exposure_id,
                )
                missing.append(item)
                missing_ids.append(item.id)
            status = cls._status(logic, matched, unmatched, unknown, len(candidate.conditions))
            assessments.append(
                ExclusionAssessment(
                    id=uuid.uuid4(),
                    coverage_assessment_id=coverage.id,
                    exposure_id=coverage.exposure_id,
                    exclusion_reference=candidate.exclusion_reference,
                    status=status,
                    rationale=cls._rationale(status, logic),
                    matched_conditions=matched,
                    unmatched_conditions=unmatched,
                    unknown_conditions=unknown,
                    policy_evidence=list(used_policy.values()),
                    claim_evidence=list(used_claim.values()),
                    missing_information_ids=missing_ids,
                )
            )
        return assessments

    @staticmethod
    def _status(
        logic: str,
        matched: list[str],
        unmatched: list[str],
        unknown: list[str],
        condition_count: int,
    ) -> Literal["applies", "does_not_apply", "indeterminate"]:
        if condition_count == 0 or logic not in {"all", "any"}:
            return "indeterminate"
        if logic == "all":
            if unmatched:
                return "does_not_apply"
            if len(matched) == condition_count and not unknown:
                return "applies"
            return "indeterminate"
        if matched:
            return "applies"
        if len(unmatched) == condition_count and not unknown:
            return "does_not_apply"
        return "indeterminate"

    @staticmethod
    def _rationale(status: str, logic: str) -> str:
        if status == "applies":
            return f"The grounded claim facts satisfy the exclusion's {logic}-condition rule."
        if status == "does_not_apply":
            return f"The grounded claim facts do not satisfy the exclusion's {logic}-condition rule."
        return "The available grounded facts or policy evidence are insufficient for a conclusion."

    @classmethod
    def _policy_evidence(
        cls, results: list[SemanticSearchResult]
    ) -> tuple[list[EvidenceRef], dict[str, EvidenceRef]]:
        evidence: list[EvidenceRef] = []
        seen: set[uuid.UUID] = set()
        for result in results:
            for item in [result, *result.context_chunks]:
                if item.chunk_id in seen:
                    continue
                seen.add(item.chunk_id)
                evidence.append(cls._evidence_from_result(f"[P{len(evidence) + 1}]", item))
        return evidence, {item.id: item for item in evidence}

    @staticmethod
    def _evidence_from_result(evidence_id: str, result: Any) -> EvidenceRef:
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
    def _related_incident_id(
        request: ExclusionAssessmentRequest, coverage: CoverageAssessment
    ) -> uuid.UUID | None:
        if coverage.exposure_id:
            for exposure in request.claim.exposures:
                if exposure.id == coverage.exposure_id:
                    return exposure.incident_id
        if len(request.claim.incidents) == 1:
            return request.claim.incidents[0].id
        return None

    @staticmethod
    def _missing(
        *,
        field_path: str,
        reason: str,
        question: str | None,
        related_incident_id: uuid.UUID | None,
        related_exposure_id: uuid.UUID | None,
    ) -> MissingInformation:
        return MissingInformation(
            id=uuid.uuid4(),
            field_path=field_path,
            reason=reason,
            required_for="exclusion_assessment",
            blocking=True,
            question=question,
            related_incident_id=related_incident_id,
            related_exposure_id=related_exposure_id,
        )

    @staticmethod
    def _bracket(value: str) -> str:
        return f"[{value}]" if re.fullmatch(r"P\d+", value) else value

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "candidate"
