"""Grounded answer generation over retrieved CaseLens evidence."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from groq import APIError, AsyncGroq
from pydantic import BaseModel, Field

from api.schemas import GroundedCitation, SemanticSearchResult


DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
_EVIDENCE_ID_RE = re.compile(r"^\[E\d+\]$")
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# The backend is the Python project root. Override inherited shell values so a
# stale process-level key cannot take precedence over Backend-ai/.env.
load_dotenv(BACKEND_ROOT / ".env", override=True)


class _LLMAnswer(BaseModel):
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool


class LLMProviderError(RuntimeError):
    """The configured LLM provider rejected or failed the request."""


@dataclass(frozen=True)
class _Evidence:
    evidence_id: str
    text: str
    citation: GroundedCitation


class GroundedAnswerService:
    def __init__(self, client=None, model: str | None = None):
        api_key = os.getenv("GROQ_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")
        self.client = client or AsyncGroq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    async def answer(
        self,
        query: str,
        results: list[SemanticSearchResult],
    ) -> tuple[_LLMAnswer, list[GroundedCitation]]:
        evidence = self._build_evidence(results)
        if not evidence:
            return (
                _LLMAnswer(
                    answer="The supplied evidence is insufficient to answer this question.",
                    evidence_ids=[],
                    insufficient_evidence=True,
                ),
                [],
            )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": self._user_prompt(query, evidence)},
                ],
            )
        except APIError as error:
            raise LLMProviderError(f"Groq request failed: {error}") from error
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty answer")
        payload = _LLMAnswer.model_validate(json.loads(content))
        payload.evidence_ids = list(
            dict.fromkeys(
                f"[{item}]" if re.fullmatch(r"E\d+", item) else item
                for item in payload.evidence_ids
            )
        )

        by_id = {item.evidence_id: item.citation for item in evidence}
        invalid = [
            item
            for item in payload.evidence_ids
            if not _EVIDENCE_ID_RE.fullmatch(item) or item not in by_id
        ]
        if invalid:
            raise ValueError(f"Groq returned unknown evidence IDs: {invalid}")
        if not payload.insufficient_evidence and not payload.evidence_ids:
            raise ValueError("A grounded answer must cite at least one supplied evidence item")
        citations = [by_id[item] for item in payload.evidence_ids]
        return payload, citations

    @staticmethod
    def _build_evidence(results: list[SemanticSearchResult]) -> list[_Evidence]:
        evidence: list[_Evidence] = []
        seen = set()

        def append(item, *, added_context: bool, anchor_chunk_id=None) -> None:
            if item.chunk_id in seen:
                return
            seen.add(item.chunk_id)
            evidence_id = f"[E{len(evidence) + 1}]"
            evidence.append(
                _Evidence(
                    evidence_id=evidence_id,
                    text=item.text,
                    citation=GroundedCitation(
                        evidence_id=evidence_id,
                        chunk_id=item.chunk_id,
                        source_file=item.source_file,
                        page_numbers=item.page_numbers,
                        section_path=item.section_path,
                        added_context=added_context,
                        anchor_chunk_id=anchor_chunk_id,
                    ),
                )
            )

        for result in results:
            append(result, added_context=False)
            for context in result.context_chunks:
                append(
                    context,
                    added_context=True,
                    anchor_chunk_id=context.anchor_chunk_id,
                )
        return evidence

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Answer only from the supplied evidence. Every sentence or bullet must be directly "
            "supported and preserve all material conditions and qualifiers. State a consequence "
            "only when the same cited evidence states it explicitly. Put distinct policy rules "
            "in separate bullets; never transfer conditions or consequences between clauses or "
            "add a combined rule. Weaken or remove any partially supported claim. Prefer fewer "
            "fully supported claims. Return JSON only: answer (string), evidence_ids (only IDs "
            "directly used), and "
            "insufficient_evidence (boolean). If evidence is inadequate, say so and set the "
            "boolean true. Answer in the question's language."
        )

    @staticmethod
    def _user_prompt(query: str, evidence: list[_Evidence]) -> str:
        blocks = [f"Question:\n{query}", "Evidence:"]
        for item in evidence:
            citation = item.citation
            blocks.append(
                "\n".join(
                    [
                        item.evidence_id,
                        f"Source: {citation.source_file}",
                        f"Pages: {citation.page_numbers}",
                        f"Section: {' > '.join(citation.section_path) or 'n/a'}",
                        item.text,
                    ]
                )
            )
        return "\n\n".join(blocks)
