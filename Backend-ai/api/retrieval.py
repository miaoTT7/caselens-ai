"""Semantic retrieval over persisted CaseLens chunks."""

from __future__ import annotations

import asyncio
import math
import os
import re
import uuid
from collections import Counter

from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import CrossEncoder

from api.ingestion import EmbeddingModel, _embedding_rows
from api.models import EMBEDDING_DIMENSION
from api.repositories import ChunkRepository, KnowledgeBaseRepository
from api.schemas import RetrievalContextChunk, SemanticSearchRequest, SemanticSearchResult


SAME_PARENT_CONTEXT_LIMIT = 6
RRF_K = 60
MIN_HYBRID_CANDIDATES = 20
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "32"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "256"))
RERANKER_WEIGHT = 0.5
RRF_WEIGHT = 0.5
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_BM25_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_default_reranker = None


class SemanticRetrievalService:
    def __init__(self, session: AsyncSession, embedding_model: EmbeddingModel, reranker_model=None):
        self.knowledge_bases = KnowledgeBaseRepository(session)
        self.chunks = ChunkRepository(session)
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model

    async def search(
        self,
        knowledge_base_id: uuid.UUID,
        request: SemanticSearchRequest,
    ) -> list[SemanticSearchResult]:
        if await self.knowledge_bases.get(knowledge_base_id) is None:
            raise LookupError("Knowledge base not found")

        query = request.query.strip()
        if not query:
            raise ValueError("Query must not be blank")
        raw_embedding = await asyncio.to_thread(self.embedding_model.encode, [query])
        embeddings = _embedding_rows(raw_embedding)
        if len(embeddings) != 1 or len(embeddings[0]) != EMBEDDING_DIMENSION:
            raise ValueError(f"Query embedding must contain {EMBEDDING_DIMENSION} values")

        candidate_limit = request.limit
        if request.retrieval_mode in {"hybrid", "hybrid_rerank"}:
            candidate_limit = max(MIN_HYBRID_CANDIDATES, request.limit * 4)
        if request.retrieval_mode == "hybrid_rerank":
            candidate_limit = max(RERANK_CANDIDATES, request.limit)

        matches = await self.chunks.search_by_vector(
            knowledge_base_id=knowledge_base_id,
            query_embedding=embeddings[0],
            limit=candidate_limit,
            document_id=request.document_id,
            source_format=request.source_format.lower() if request.source_format else None,
            page_number=request.page_number,
            section=request.section,
            sheet=request.sheet,
        )
        if request.retrieval_mode in {"hybrid", "hybrid_rerank"}:
            corpus = await self.chunks.list_searchable(
                knowledge_base_id=knowledge_base_id,
                document_id=request.document_id,
                source_format=request.source_format.lower() if request.source_format else None,
                page_number=request.page_number,
                section=request.section,
                sheet=request.sheet,
            )
            bm25_matches = self._bm25_search(query, corpus, candidate_limit)
            fused_limit = RERANK_CANDIDATES if request.retrieval_mode == "hybrid_rerank" else request.limit
            anchors, results = self._fuse(matches, bm25_matches, fused_limit)
        else:
            anchors = [chunk for chunk, _ in matches]
            results = [
                self._result(chunk, distance, rank)
                for rank, (chunk, distance) in enumerate(matches, start=1)
            ]
        await self._expand_context(anchors, results)
        if request.retrieval_mode == "hybrid_rerank":
            anchors, results = await self._rerank(query, anchors, results, request.limit)
        return results

    async def _expand_context(self, anchors, results: list[SemanticSearchResult]) -> None:
        original_ids = {chunk.id for chunk in anchors}
        added_ids: set[uuid.UUID] = set()
        document_chunks: dict[uuid.UUID, list] = {}

        for anchor, result in zip(anchors, results, strict=True):
            if anchor.document_id not in document_chunks:
                document_chunks[anchor.document_id] = list(
                    await self.chunks.list_for_document(anchor.document_id)
                )
            candidates = document_chunks[anchor.document_id]
            by_order = {chunk.chunk_order: chunk for chunk in candidates}
            reasons: dict[uuid.UUID, set[str]] = {}
            selected: dict[uuid.UUID, object] = {}

            for offset, reason in ((-1, "adjacent_previous"), (1, "adjacent_next")):
                candidate = by_order.get(anchor.chunk_order + offset)
                if candidate is not None and self._sections_related(anchor.section_path, candidate.section_path):
                    selected[candidate.id] = candidate
                    reasons.setdefault(candidate.id, set()).add(reason)

            parent = self._parent_section(anchor.section_path)
            if parent:
                siblings = [
                    chunk
                    for chunk in candidates
                    if chunk.id != anchor.id and self._parent_section(chunk.section_path) == parent
                ]
                # Expand a complete, small subsection family. For broad parents
                # such as "Contents" or "General conditions", partial sibling
                # expansion would add arbitrary, unrelated sections; adjacent
                # expansion above remains available for those parents.
                if len(siblings) <= SAME_PARENT_CONTEXT_LIMIT:
                    siblings.sort(key=lambda chunk: chunk.chunk_order)
                    for candidate in siblings:
                        selected[candidate.id] = candidate
                        reasons.setdefault(candidate.id, set()).add("same_parent_section")

            for candidate in sorted(selected.values(), key=lambda chunk: chunk.chunk_order):
                if candidate.id in original_ids or candidate.id in added_ids:
                    continue
                result.context_chunks.append(
                    self._context_result(candidate, anchor.id, sorted(reasons[candidate.id]))
                )
                added_ids.add(candidate.id)

    @staticmethod
    def _parent_section(section_path: list[str]) -> tuple[str, ...]:
        return tuple(section_path[:-1]) if len(section_path) > 1 else ()

    @classmethod
    def _sections_related(cls, first: list[str], second: list[str]) -> bool:
        if not first or not second:
            return False
        if first == second:
            return True
        first_parent = cls._parent_section(first)
        second_parent = cls._parent_section(second)
        if first_parent and first_parent == second_parent:
            return True
        shorter, longer = (first, second) if len(first) <= len(second) else (second, first)
        return longer[: len(shorter)] == shorter

    @staticmethod
    def _bm25_tokens(text: str) -> list[str]:
        def normalize(token: str) -> str:
            token = token.casefold()
            if len(token) > 4 and token.endswith("ies"):
                return token[:-3] + "y"
            if len(token) > 4 and token.endswith("sses"):
                return token[:-2]
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                return token[:-1]
            return token

        return [normalize(token) for token in _BM25_TOKEN_RE.findall(text or "")]

    @classmethod
    def _bm25_search(cls, query: str, corpus, limit: int):
        query_tokens = cls._bm25_tokens(query)
        if not query_tokens or not corpus:
            return []
        tokenized = [cls._bm25_tokens(chunk.text) for chunk in corpus]
        average_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
        document_frequency = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        scores = []
        total = len(tokenized)
        k1, b = 1.5, 0.75
        for chunk, tokens in zip(corpus, tokenized, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                df = document_frequency[token]
                inverse_document_frequency = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(tokens) / average_length)
                score += inverse_document_frequency * frequency * (k1 + 1) / denominator
            if score > 0:
                scores.append((chunk, score))
        scores.sort(key=lambda item: (-item[1], item[0].chunk_order, str(item[0].id)))
        return scores[:limit]

    @classmethod
    def _fuse(cls, vector_matches, bm25_matches, limit: int):
        fused: dict[uuid.UUID, dict] = {}
        for rank, (chunk, distance) in enumerate(vector_matches, start=1):
            fused[chunk.id] = {
                "chunk": chunk,
                "vector_rank": rank,
                "vector_score": max(-1.0, min(1.0, 1.0 - float(distance))),
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 1 / (RRF_K + rank),
            }
        for rank, (chunk, score) in enumerate(bm25_matches, start=1):
            item = fused.setdefault(
                chunk.id,
                {
                    "chunk": chunk,
                    "vector_rank": None,
                    "vector_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                },
            )
            item["bm25_rank"] = rank
            item["bm25_score"] = float(score)
            item["rrf_score"] += 1 / (RRF_K + rank)
        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -item["rrf_score"],
                item["vector_rank"] or math.inf,
                item["bm25_rank"] or math.inf,
                str(item["chunk"].id),
            ),
        )[:limit]
        anchors = [item["chunk"] for item in ranked]
        results = [
            cls._hybrid_result(item, rank)
            for rank, item in enumerate(ranked, start=1)
        ]
        return anchors, results

    async def _rerank(self, query: str, anchors, results: list[SemanticSearchResult], limit: int):
        model = self.reranker_model or await asyncio.to_thread(self._default_reranker_model)
        # Context remains attached to the returned result for answer generation,
        # while the calibrated BGE reranker scores only the anchor chunk.
        documents = [result.text for result in results]
        pairs = [[query, document] for document in documents]
        raw_scores = await asyncio.to_thread(model.predict, pairs, batch_size=RERANK_BATCH_SIZE)
        reranker_scores = self._normalize_scores(raw_scores)
        rrf_scores = self._min_max_scores([result.rrf_score or 0.0 for result in results])

        combined = []
        for anchor, result, reranker_score, rrf_score in zip(
            anchors, results, reranker_scores, rrf_scores, strict=True
        ):
            result.pre_rerank_rank = result.rank
            result.reranker_score = float(reranker_score)
            result.retrieval_method = "hybrid_rerank"
            result.score = RERANKER_WEIGHT * float(reranker_score) + RRF_WEIGHT * float(rrf_score)
            combined.append((anchor, result))
        combined.sort(key=lambda item: (-item[1].score, item[1].pre_rerank_rank or math.inf))
        combined = combined[:limit]
        for rank, (_, result) in enumerate(combined, start=1):
            result.rank = rank
        return [anchor for anchor, _ in combined], [result for _, result in combined]

    @staticmethod
    def _normalize_scores(values) -> list[float]:
        scores = [float(value) for value in values]
        if not scores:
            return []
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("Reranker returned a non-finite relevance score")
        minimum, maximum = min(scores), max(scores)
        if minimum >= 0.0 and maximum <= 1.0:
            return scores
        spread = maximum - minimum
        if spread < 1e-3:
            return [min(1.0, max(0.0, score)) for score in scores]
        return [(score - minimum) / spread for score in scores]

    @staticmethod
    def _min_max_scores(values) -> list[float]:
        scores = [float(value) for value in values]
        if not scores:
            return []
        minimum, maximum = min(scores), max(scores)
        spread = maximum - minimum
        if spread <= 0:
            return [0.0 for _ in scores]
        return [(score - minimum) / spread for score in scores]

    @staticmethod
    def _default_reranker_model():
        global _default_reranker
        if _default_reranker is None:
            model_name = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
            _default_reranker = CrossEncoder(
                model_name,
                max_length=RERANK_MAX_LENGTH,
                device="cpu",
            )
        return _default_reranker

    @staticmethod
    def _result(chunk, distance: float, rank: int) -> SemanticSearchResult:
        similarity = max(-1.0, min(1.0, 1.0 - distance))
        return SemanticSearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=similarity,
            chunk_type=chunk.chunk_type,
            source_file=chunk.source_file,
            source_format=chunk.source_format,
            page_numbers=chunk.page_numbers,
            section_path=chunk.section_path,
            sheet=chunk.sheet,
            row_start=chunk.row_start,
            row_end=chunk.row_end,
            email_metadata=chunk.email_metadata,
            rank=rank,
            vector_rank=rank,
            vector_score=similarity,
        )

    @staticmethod
    def _hybrid_result(item: dict, rank: int) -> SemanticSearchResult:
        chunk = item["chunk"]
        return SemanticSearchResult(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            score=item["rrf_score"],
            chunk_type=chunk.chunk_type,
            source_file=chunk.source_file,
            source_format=chunk.source_format,
            page_numbers=chunk.page_numbers,
            section_path=chunk.section_path,
            sheet=chunk.sheet,
            row_start=chunk.row_start,
            row_end=chunk.row_end,
            email_metadata=chunk.email_metadata,
            rank=rank,
            retrieval_method="hybrid",
            vector_rank=item["vector_rank"],
            vector_score=item["vector_score"],
            bm25_rank=item["bm25_rank"],
            bm25_score=item["bm25_score"],
            rrf_score=item["rrf_score"],
        )

    @staticmethod
    def _context_result(chunk, anchor_chunk_id: uuid.UUID, reasons: list[str]) -> RetrievalContextChunk:
        return RetrievalContextChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            chunk_type=chunk.chunk_type,
            source_file=chunk.source_file,
            source_format=chunk.source_format,
            page_numbers=chunk.page_numbers,
            section_path=chunk.section_path,
            sheet=chunk.sheet,
            row_start=chunk.row_start,
            row_end=chunk.row_end,
            email_metadata=chunk.email_metadata,
            context_reasons=reasons,
            anchor_chunk_id=anchor_chunk_id,
        )
