"""Phase 1 extraction of explicitly stated claim facts."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from groq import APIError
from pydantic import AliasChoices, BaseModel, Field, model_validator

from api.claim_schemas import (
    Claim,
    ClaimFact,
    ClaimFactExtractionRequest,
    ClaimFactExtractionResponse,
    EvidenceRef,
    Incident,
)
from api.llm_service import GroundedAnswerService, LLMProviderError


FACT_PATHS = (
    "claim.claim_number",
    "claim.reported_date",
    "claim.claimant_reference",
    "claim.description",
    "incidents[0].incident_type",
    "incidents[0].event_date",
    "incidents[0].location",
    "incidents[0].cause",
    "incidents[0].description",
    "claim.claimed_amount",
    "claim.currency",
)
DATETIME_PATHS = {"claim.reported_date", "incidents[0].event_date"}


class _ExtractedEvidence(BaseModel):
    source_id: str
    text_quote: str = Field(validation_alias=AliasChoices("text_quote", "quote"))


class _ExtractedFact(BaseModel):
    fact_path: str
    value: Any | None = Field(validation_alias=AliasChoices("value", "normalized_value"))
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[_ExtractedEvidence] = Field(default_factory=list)


class _ExtractionPayload(BaseModel):
    facts: list[_ExtractedFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def include_every_allowed_path(self):
        present = {fact.fact_path for fact in self.facts}
        self.facts.extend(
            _ExtractedFact(fact_path=path, value=None)
            for path in FACT_PATHS
            if path not in present
        )
        return self


class ClaimFactExtractionService:
    def __init__(self, client=None, model: str | None = None):
        configured = GroundedAnswerService(client=client, model=model)
        self.client = configured.client
        self.model = configured.model

    async def extract(self, request: ClaimFactExtractionRequest) -> ClaimFactExtractionResponse:
        sources = self._sources(request)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._user_prompt(sources)},
                ],
            )
        except APIError as error:
            raise LLMProviderError(f"Groq request failed: {error}") from error
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty extraction")
        payload = _ExtractionPayload.model_validate(json.loads(content))
        return self._build_response(payload, sources)

    @staticmethod
    def _sources(request: ClaimFactExtractionRequest) -> dict[str, dict[str, str]]:
        sources: dict[str, dict[str, str]] = {}
        if request.fnol_text and request.fnol_text.strip():
            sources["[S1]"] = {"text": request.fnol_text.strip(), "source_file": "fnol_text"}
        if request.parsed_document_text and request.parsed_document_text.strip():
            source_id = f"[S{len(sources) + 1}]"
            sources[source_id] = {
                "text": request.parsed_document_text.strip(),
                "source_file": request.parsed_document_name or "parsed_claim_document",
            }
        return sources

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Extract only facts explicitly stated in the supplied claim sources. Never infer, "
            "guess, calculate, or fill missing values. Return JSON with a facts array. Each fact "
            "must use exactly these keys: fact_path, value, confidence, evidence. Each evidence item "
            "must use exactly these keys: source_id, text_quote. Copy source_id exactly as shown, "
            "including square brackets. Return supported facts only. Dates must use ISO 8601; "
            "amounts must contain digits only with an optional decimal point."
        )

    @staticmethod
    def _user_prompt(sources: dict[str, dict[str, str]]) -> str:
        fields = "\n".join(f"- {path}" for path in FACT_PATHS)
        blocks = ["Allowed fact paths:", fields, "Claim sources:"]
        for source_id, source in sources.items():
            blocks.append(f"{source_id} ({source['source_file']}):\n{source['text']}")
        return "\n\n".join(blocks)

    @classmethod
    def _build_response(
        cls,
        payload: _ExtractionPayload,
        sources: dict[str, dict[str, str]],
    ) -> ClaimFactExtractionResponse:
        accepted: dict[str, tuple[Any, float, list[EvidenceRef]]] = {}
        evidence_registry: list[EvidenceRef] = []

        for candidate in payload.facts:
            if candidate.fact_path not in FACT_PATHS or candidate.value is None:
                continue
            evidence = cls._validated_evidence(candidate.evidence, sources, evidence_registry)
            if not evidence:
                continue
            value = cls._normalize_value(candidate.fact_path, candidate.value)
            if value is None:
                continue
            accepted[candidate.fact_path] = (value, candidate.confidence, evidence)

        for candidate in cls._deterministic_candidates(sources):
            if candidate.fact_path in accepted:
                continue
            evidence = cls._validated_evidence(candidate.evidence, sources, evidence_registry)
            value = cls._normalize_value(candidate.fact_path, candidate.value)
            if evidence and value is not None:
                accepted[candidate.fact_path] = (value, candidate.confidence, evidence)

        facts = []
        for path in FACT_PATHS:
            if path in accepted:
                value, confidence, evidence = accepted[path]
                facts.append(
                    ClaimFact(
                        id=uuid.uuid4(),
                        fact_path=path,
                        value=value,
                        status="extracted",
                        confidence=confidence,
                        claim_evidence=evidence,
                    )
                )
            else:
                facts.append(
                    ClaimFact(
                        id=uuid.uuid4(),
                        fact_path=path,
                        value=None,
                        status="unknown",
                        confidence=None,
                    )
                )

        values = {fact.fact_path: fact.value for fact in facts}
        incident = Incident(
            id=uuid.uuid4(),
            incident_type=values["incidents[0].incident_type"],
            event_date=values["incidents[0].event_date"],
            location=values["incidents[0].location"],
            cause=values["incidents[0].cause"],
            description=values["incidents[0].description"],
        )
        claim = Claim(
            id=uuid.uuid4(),
            claim_number=values["claim.claim_number"],
            reported_date=values["claim.reported_date"],
            claimant_reference=values["claim.claimant_reference"],
            description=values["claim.description"],
            incidents=[incident],
        )
        return ClaimFactExtractionResponse(
            claim=claim,
            facts=facts,
            claim_evidence=evidence_registry,
        )

    @classmethod
    def _validated_evidence(
        cls,
        candidates: list[_ExtractedEvidence],
        sources: dict[str, dict[str, str]],
        registry: list[EvidenceRef],
    ) -> list[EvidenceRef]:
        validated = []
        for candidate in candidates:
            source_id = cls._canonical_source_id(candidate.source_id)
            source = sources.get(source_id)
            quote = candidate.text_quote.strip()
            if not source or not quote or cls._compact(quote) not in cls._compact(source["text"]):
                continue
            evidence = EvidenceRef(
                id=f"[C{len(registry) + 1}]",
                evidence_type="claim",
                source_file=source["source_file"],
                text_quote=quote,
                metadata={"source_id": source_id},
            )
            registry.append(evidence)
            validated.append(evidence)
        return validated

    @staticmethod
    def _canonical_source_id(value: str) -> str:
        source_id = value.strip()
        return source_id if source_id.startswith("[") and source_id.endswith("]") else f"[{source_id}]"

    @classmethod
    def _deterministic_candidates(
        cls, sources: dict[str, dict[str, str]]
    ) -> list[_ExtractedFact]:
        candidates: list[_ExtractedFact] = []
        month_names = (
            "January|February|March|April|May|June|July|August|September|October|November|December"
        )
        date_pattern = re.compile(rf"\b(\d{{1,2}}\s+(?:{month_names})\s+\d{{4}})\b", re.IGNORECASE)
        money_pattern = re.compile(
            r"\b(CHF|EUR|USD|GBP)\s*([0-9][0-9' ,]*(?:\.[0-9]+)?)\b", re.IGNORECASE
        )
        location_pattern = re.compile(
            rf"\b((?:outside|at|near|in)\s+[^.,]+?)(?=\s+on\s+\d{{1,2}}\s+(?:{month_names})\s+\d{{4}}|[.,]|$)",
            re.IGNORECASE,
        )
        theft_pattern = re.compile(r"\b(stolen|theft)\b", re.IGNORECASE)
        for source_id, source in sources.items():
            text = source["text"]
            if match := theft_pattern.search(text):
                candidates.append(cls._candidate(
                    "incidents[0].incident_type", "theft", source_id, match.group(0)
                ))
            if match := date_pattern.search(text):
                try:
                    normalized = datetime.strptime(match.group(1), "%d %B %Y").date().isoformat()
                    candidates.append(cls._candidate(
                        "incidents[0].event_date", normalized, source_id, match.group(1)
                    ))
                except ValueError:
                    pass
            if match := money_pattern.search(text):
                amount = re.sub(r"[' ,]", "", match.group(2))
                candidates.extend([
                    cls._candidate("claim.claimed_amount", amount, source_id, match.group(0)),
                    cls._candidate("claim.currency", match.group(1).upper(), source_id, match.group(0)),
                ])
            if match := location_pattern.search(text):
                candidates.append(cls._candidate(
                    "incidents[0].location", match.group(1).strip(), source_id, match.group(1).strip()
                ))
        return candidates

    @staticmethod
    def _candidate(path: str, value: Any, source_id: str, quote: str) -> _ExtractedFact:
        return _ExtractedFact(
            fact_path=path,
            value=value,
            confidence=1.0,
            evidence=[_ExtractedEvidence(source_id=source_id, text_quote=quote)],
        )

    @staticmethod
    def _compact(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _normalize_value(path: str, value: Any) -> Any | None:
        if path in DATETIME_PATHS:
            if not isinstance(value, str):
                return None
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if path == "claim.claimed_amount":
            try:
                return Decimal(str(value))
            except (InvalidOperation, ValueError):
                return None
        if path == "claim.currency":
            return str(value).upper().strip() or None
        return str(value).strip() or None
