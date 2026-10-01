"""Phase 5 policyholder-obligation assessment."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Literal

from groq import APIError
from pydantic import BaseModel, Field

from api.claim_schemas import (
    ClaimFact,
    CoverageAssessment,
    EvidenceRef,
    MissingInformation,
    ObligationAssessment,
    ObligationAssessmentRequest,
    ObligationAssessmentResponse,
)
from api.llm_service import GroundedAnswerService, LLMProviderError
from api.retrieval import SemanticRetrievalService
from api.schemas import SemanticSearchRequest, SemanticSearchResult


class _ObligationCondition(BaseModel):
    description: str
    result: str
    fact_paths: list[str] = Field(default_factory=list)
    policy_evidence_ids: list[str] = Field(default_factory=list)


class _ObligationCandidate(BaseModel):
    obligation_reference: str
    conditions: list[_ObligationCondition] = Field(default_factory=list)
    requires_culpable_breach: bool | None = None
    culpable_breach: bool | None = None
    culpability_fact_paths: list[str] = Field(default_factory=list)
    culpability_policy_evidence_ids: list[str] = Field(default_factory=list)
    requires_effect_on_loss: bool | None = None
    effect_on_loss: str | None = None
    effect_fact_paths: list[str] = Field(default_factory=list)
    effect_policy_evidence_ids: list[str] = Field(default_factory=list)
    permitted_consequence: str | None = None
    consequence_policy_evidence_ids: list[str] = Field(default_factory=list)


class _ObligationPayload(BaseModel):
    obligations: list[_ObligationCandidate] = Field(default_factory=list)


class ObligationAssessmentService:
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

    async def assess(self, request: ObligationAssessmentRequest) -> ObligationAssessmentResponse:
        fact_map = {
            fact.fact_path: fact
            for fact in request.facts
            if fact.value is not None and fact.status != "unknown"
        }
        assessments: list[ObligationAssessment] = []
        missing: list[MissingInformation] = []

        for coverage in request.coverage_assessments:
            if coverage.status == "not_covered":
                continue
            document_id = self._coverage_document_id(coverage)
            if document_id is None:
                missing.append(
                    self._missing(
                        field_path="claim.policy.document_id",
                        reason=(
                            "A unique applicable policy document is required before policyholder "
                            f"obligations can be assessed for {coverage.coverage_reference}."
                        ),
                        question="Which policy document and version apply to this claim?",
                        related_incident_id=self._related_incident_id(request, coverage),
                        related_exposure_id=coverage.exposure_id,
                    )
                )
                continue

            results = await self.retrieval.search(
                request.knowledge_base_id,
                SemanticSearchRequest(
                    query=self._obligation_query(request, coverage, fact_map),
                    limit=request.retrieval_limit,
                    retrieval_mode="hybrid_rerank",
                    document_id=document_id,
                ),
            )
            evidence, evidence_by_id = self._policy_evidence(results)
            try:
                payload = await self._extract_obligations(coverage, fact_map, evidence)
            except LLMProviderError:
                missing.append(
                    self._missing(
                        field_path="obligations.provider_assessment",
                        reason=(
                            "Policyholder obligations could not be assessed because the LLM "
                            "provider was unavailable. Human review is required."
                        ),
                        question=None,
                        related_incident_id=self._related_incident_id(request, coverage),
                        related_exposure_id=coverage.exposure_id,
                    ).model_copy(update={"metadata": {"assessment_unavailable": True}})
                )
                return ObligationAssessmentResponse(
                    claim_id=request.claim.id,
                    status="unavailable",
                    error="Obligation assessment is unavailable due to an LLM provider failure.",
                    obligation_assessments=[],
                    missing_information=missing,
                )
            coverage_results = self._build_assessments(
                request, coverage, payload, fact_map, evidence_by_id, missing
            )
            assessments.extend(coverage_results)
            if not coverage_results:
                missing.append(
                    self._missing(
                        field_path=f"obligations.{self._slug(coverage.coverage_reference)}",
                        reason=(
                            "No relevant policyholder obligation could be established from the "
                            f"retrieved policy evidence for {coverage.coverage_reference}."
                        ),
                        question=None,
                        related_incident_id=self._related_incident_id(request, coverage),
                        related_exposure_id=coverage.exposure_id,
                    )
                )

        return ObligationAssessmentResponse(
            claim_id=request.claim.id,
            obligation_assessments=assessments,
            missing_information=missing,
        )

    @staticmethod
    def _coverage_document_id(coverage: CoverageAssessment) -> uuid.UUID | None:
        document_ids = {
            item.document_id for item in coverage.policy_evidence if item.document_id is not None
        }
        return next(iter(document_ids)) if len(document_ids) == 1 else None

    @staticmethod
    def _obligation_query(
        request: ObligationAssessmentRequest,
        coverage: CoverageAssessment,
        facts: dict[str, ClaimFact],
    ) -> str:
        incidents = "; ".join(
            f"type={item.incident_type}, event_date={item.event_date}, cause={item.cause}, "
            f"location={item.location}"
            for item in request.claim.incidents
        )
        fact_text = "; ".join(f"{path}={fact.value}" for path, fact in facts.items())
        exclusions = "; ".join(
            f"{item.exclusion_reference}={item.status}"
            for item in request.exclusion_assessments
            if item.coverage_assessment_id == coverage.id
        )
        return (
            f"Find policyholder obligations relevant to coverage '{coverage.coverage_reference}' and "
            f"this incident. Incidents: {incidents or 'none supplied'}. Explicit claim facts: "
            f"{fact_text or 'none supplied'}. Relevant exclusion assessments: "
            f"{exclusions or 'none'}. Search for notification, reporting, loss mitigation, document "
            "provision, police reporting, cooperation, and similar duties. Ignore deductibles, limits, "
            "other insurance calculations, payable amount, and final claim decisions."
        )

    async def _extract_obligations(
        self,
        coverage: CoverageAssessment,
        facts: dict[str, ClaimFact],
        evidence: list[EvidenceRef],
    ) -> _ObligationPayload:
        fact_text = "\n".join(f"{path}: {fact.value}" for path, fact in facts.items())
        evidence_text = "\n\n".join(
            f"{item.id}\nSource: {item.source_file}\n"
            f"Section: {' > '.join(item.section_path)}\n{item.text_quote}"
            for item in evidence
        )
        prompt = (
            f"Coverage: {coverage.coverage_reference}\n"
            "Return exactly one JSON object and nothing else: {\"obligations\":[{"
            "\"obligation_reference\":\"...\",\"conditions\":[{\"description\":\"...\","
            "\"result\":\"matched|unmatched|unknown\",\"fact_paths\":[],"
            "\"policy_evidence_ids\":[]}],\"requires_culpable_breach\":null,"
            "\"culpable_breach\":null,\"culpability_fact_paths\":[],"
            "\"culpability_policy_evidence_ids\":[],\"requires_effect_on_loss\":null,"
            "\"effect_on_loss\":\"unknown\",\"effect_fact_paths\":[],"
            "\"effect_policy_evidence_ids\":[],\"permitted_consequence\":null,"
            "\"consequence_policy_evidence_ids\":[]}]}. obligations[] may contain JSON objects only. "
            "Always include every key. Use unknown for enum-like unknowns; use null only for nullable "
            "boolean/text fields. Use only supplied facts and evidence. Do not infer consequences.\n\n"
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
                            "Extract grounded policyholder obligations. JSON only; no markdown or explanation."
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
            raise ValueError("Groq returned an empty obligation assessment")
        return _ObligationPayload.model_validate(json.loads(content))

    @staticmethod
    def _is_json_generation_error(error: APIError) -> bool:
        return getattr(error, "status_code", None) == 400 and "json_validate_failed" in str(error)

    @classmethod
    def _build_assessments(
        cls,
        request: ObligationAssessmentRequest,
        coverage: CoverageAssessment,
        payload: _ObligationPayload,
        facts: dict[str, ClaimFact],
        policy_evidence: dict[str, EvidenceRef],
        missing: list[MissingInformation],
    ) -> list[ObligationAssessment]:
        assessments: list[ObligationAssessment] = []
        for candidate in payload.obligations:
            matched: list[str] = []
            unmatched: list[str] = []
            unknown: list[str] = []
            used_policy: dict[str, EvidenceRef] = {}
            used_claim: dict[str, EvidenceRef] = {}
            missing_ids: list[uuid.UUID] = []

            for index, condition in enumerate(candidate.conditions):
                result, valid_policy = cls._validated_result(
                    condition.result,
                    condition.fact_paths,
                    condition.policy_evidence_ids,
                    facts,
                    policy_evidence,
                )
                cls._collect_policy(valid_policy, used_policy)
                cls._collect_claim(condition.fact_paths, facts, used_claim)
                if result == "matched":
                    matched.append(condition.description)
                elif result == "unmatched":
                    unmatched.append(condition.description)
                else:
                    unknown.append(condition.description)
                    missing_paths = [path for path in condition.fact_paths if path not in facts]
                    field_path = missing_paths[0] if missing_paths else (
                        condition.fact_paths[0] if condition.fact_paths else
                        f"obligations.{cls._slug(candidate.obligation_reference)}.condition_{index + 1}"
                    )
                    item = cls._missing(
                        field_path=field_path,
                        reason=f"Information is needed to assess obligation: {condition.description}",
                        question=f"Please provide information for: {condition.description}",
                        related_incident_id=cls._related_incident_id(request, coverage),
                        related_exposure_id=coverage.exposure_id,
                    )
                    missing.append(item)
                    missing_ids.append(item.id)

            status = cls._status(matched, unmatched, unknown, len(candidate.conditions))
            if not candidate.conditions:
                item = cls._missing(
                    field_path=f"obligations.{cls._slug(candidate.obligation_reference)}.conditions",
                    reason="No grounded conditions were available to assess this obligation.",
                    question=None,
                    related_incident_id=cls._related_incident_id(request, coverage),
                    related_exposure_id=coverage.exposure_id,
                )
                missing.append(item)
                missing_ids.append(item.id)

            culpable_breach = None
            effect_on_loss = "unknown"
            permitted_consequence = None
            if status == "breached":
                culpable_breach = cls._supported_culpability(
                    request, coverage, candidate, facts, policy_evidence,
                    used_policy, used_claim, missing, missing_ids,
                )
                effect_on_loss = cls._supported_effect(
                    request, coverage, candidate, facts, policy_evidence,
                    used_policy, used_claim, missing, missing_ids,
                )
                permitted_consequence = cls._supported_consequence(
                    candidate, policy_evidence, used_policy
                )

            assessments.append(
                ObligationAssessment(
                    id=uuid.uuid4(),
                    coverage_assessment_id=coverage.id,
                    exposure_id=coverage.exposure_id,
                    obligation_reference=candidate.obligation_reference,
                    status=status,
                    culpable_breach=culpable_breach,
                    effect_on_loss=effect_on_loss,
                    permitted_consequence=permitted_consequence,
                    rationale=cls._rationale(status),
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
    def _validated_result(
        cls,
        result: str,
        fact_paths: list[str],
        evidence_ids: list[str],
        facts: dict[str, ClaimFact],
        policy_evidence: dict[str, EvidenceRef],
    ) -> tuple[str, list[EvidenceRef]]:
        valid_policy = [
            policy_evidence[item]
            for item in (cls._bracket(value) for value in evidence_ids)
            if item in policy_evidence
        ]
        normalized = result.casefold()
        if (
            not valid_policy
            or not fact_paths
            or any(path not in facts for path in fact_paths)
            or normalized not in {"matched", "unmatched", "unknown"}
        ):
            normalized = "unknown"
        return normalized, valid_policy

    @staticmethod
    def _status(
        matched: list[str], unmatched: list[str], unknown: list[str], condition_count: int
    ) -> Literal["satisfied", "breached", "indeterminate"]:
        if condition_count == 0:
            return "indeterminate"
        if unmatched:
            return "breached"
        if len(matched) == condition_count and not unknown:
            return "satisfied"
        return "indeterminate"

    @classmethod
    def _supported_culpability(
        cls, request, coverage, candidate, facts, policy_evidence,
        used_policy, used_claim, missing, missing_ids,
    ) -> bool | None:
        if candidate.requires_culpable_breach is not True:
            return None
        value, valid_policy, valid_facts = cls._validate_supported_value(
            candidate.culpable_breach,
            candidate.culpability_fact_paths,
            candidate.culpability_policy_evidence_ids,
            facts,
            policy_evidence,
            {True, False},
        )
        cls._collect_policy(valid_policy, used_policy)
        cls._collect_claim(valid_facts, facts, used_claim)
        if value is not None:
            return value
        missing_path = next(
            (path for path in candidate.culpability_fact_paths if path not in facts),
            "obligation.culpable_breach",
        )
        item = cls._missing(
            field_path=missing_path,
            reason="The policy requires culpability, but the available facts do not establish it.",
            question="Was the failure to comply intentional or negligent?",
            related_incident_id=cls._related_incident_id(request, coverage),
            related_exposure_id=coverage.exposure_id,
        )
        missing.append(item)
        missing_ids.append(item.id)
        return None

    @classmethod
    def _supported_effect(
        cls, request, coverage, candidate, facts, policy_evidence,
        used_policy, used_claim, missing, missing_ids,
    ) -> Literal["affected", "not_affected", "unknown"]:
        if candidate.requires_effect_on_loss is not True:
            return "unknown"
        value, valid_policy, valid_facts = cls._validate_supported_value(
            candidate.effect_on_loss,
            candidate.effect_fact_paths,
            candidate.effect_policy_evidence_ids,
            facts,
            policy_evidence,
            {"affected", "not_affected"},
        )
        cls._collect_policy(valid_policy, used_policy)
        cls._collect_claim(valid_facts, facts, used_claim)
        if value in {"affected", "not_affected"}:
            return value
        missing_path = next(
            (path for path in candidate.effect_fact_paths if path not in facts),
            "obligation.effect_on_loss",
        )
        item = cls._missing(
            field_path=missing_path,
            reason=(
                "The policy requires an effect on the loss, but the available facts do not establish it."
            ),
            question="Did the failure to comply affect the occurrence or amount of the loss?",
            related_incident_id=cls._related_incident_id(request, coverage),
            related_exposure_id=coverage.exposure_id,
        )
        missing.append(item)
        missing_ids.append(item.id)
        return "unknown"

    @classmethod
    def _validate_supported_value(
        cls, value, fact_paths, evidence_ids, facts, policy_evidence, allowed_values,
    ):
        valid_policy = [
            policy_evidence[item]
            for item in (cls._bracket(raw) for raw in evidence_ids)
            if item in policy_evidence
        ]
        valid_facts = [path for path in fact_paths if path in facts]
        if (
            value not in allowed_values
            or not valid_policy
            or not fact_paths
            or len(valid_facts) != len(fact_paths)
        ):
            return None, valid_policy, valid_facts
        return value, valid_policy, valid_facts

    @classmethod
    def _supported_consequence(
        cls, candidate, policy_evidence, used_policy
    ) -> str | None:
        if not candidate.permitted_consequence:
            return None
        valid_policy = [
            policy_evidence[item]
            for item in (cls._bracket(raw) for raw in candidate.consequence_policy_evidence_ids)
            if item in policy_evidence
        ]
        if not valid_policy:
            return None
        cls._collect_policy(valid_policy, used_policy)
        return candidate.permitted_consequence

    @staticmethod
    def _collect_policy(items, destination) -> None:
        for item in items:
            destination[item.id] = item

    @staticmethod
    def _collect_claim(paths, facts, destination) -> None:
        for path in paths:
            fact = facts.get(path)
            if fact:
                for item in fact.claim_evidence:
                    destination[item.id] = item

    @staticmethod
    def _rationale(status: str) -> str:
        if status == "satisfied":
            return "The grounded claim facts show that all identified obligation conditions were met."
        if status == "breached":
            return "The grounded claim facts show that at least one required obligation condition was not met."
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
    def _related_incident_id(request, coverage) -> uuid.UUID | None:
        if coverage.exposure_id:
            for exposure in request.claim.exposures:
                if exposure.id == coverage.exposure_id:
                    return exposure.incident_id
        if len(request.claim.incidents) == 1:
            return request.claim.incidents[0].id
        return None

    @staticmethod
    def _missing(
        *, field_path, reason, question, related_incident_id, related_exposure_id,
    ) -> MissingInformation:
        return MissingInformation(
            id=uuid.uuid4(),
            field_path=field_path,
            reason=reason,
            required_for="obligation_assessment",
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
