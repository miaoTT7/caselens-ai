"""Run read-only Top-20 diagnostics against the current pgvector corpus."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(PROJECT_DIR / ".env")

from api.database import close_database_connection, get_session_factory  # noqa: E402
from api.ingestion import _embedding_rows  # noqa: E402
from api.models import EMBEDDING_DIMENSION  # noqa: E402
from api.repositories import ChunkRepository, DocumentRepository, KnowledgeBaseRepository  # noqa: E402


EVALUATION_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALUATION_DIR / "dataset.json"
NOTES_PATH = EVALUATION_DIR / "diagnostic_notes.json"
OUTPUT_DIR = EVALUATION_DIR / "results"
MODEL_NAME = "all-MiniLM-L6-v2"


def similarity(distance: float) -> float:
    return max(-1.0, min(1.0, 1.0 - distance))


def chunk_payload(chunk, score: float, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": str(chunk.id),
        "score": score,
        "document_id": str(chunk.document_id),
        "source_file": chunk.source_file,
        "token_count": chunk.token_count,
        "page_numbers": chunk.page_numbers,
        "section_path": chunk.section_path,
        "text": chunk.text,
    }


async def run() -> dict:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)
    async with get_session_factory()() as session:
        knowledge_base = await KnowledgeBaseRepository(session).get_by_name(dataset["knowledge_base"])
        if knowledge_base is None:
            raise RuntimeError(f"Knowledge base not found: {dataset['knowledge_base']}")

        documents = await DocumentRepository(session).list_for_knowledge_base(knowledge_base.id)
        chunk_repository = ChunkRepository(session)
        corpus_size = 0
        for document in documents:
            corpus_size += len(await chunk_repository.list_for_document(document.id))
        rows = []

        for item in dataset["queries"]:
            started = perf_counter()
            encoded = await asyncio.to_thread(model.encode, [item["query"].strip()])
            embeddings = _embedding_rows(encoded)
            if len(embeddings) != 1 or len(embeddings[0]) != EMBEDDING_DIMENSION:
                raise ValueError(f"Query embedding must contain {EMBEDDING_DIMENSION} values")

            matches = await chunk_repository.search_by_vector(
                knowledge_base_id=knowledge_base.id,
                query_embedding=embeddings[0],
                limit=corpus_size,
            )
            latency_ms = (perf_counter() - started) * 1000
            expected_ids = set(item["expected_chunk_ids"])
            ranked = [
                chunk_payload(chunk, similarity(distance), rank)
                for rank, (chunk, distance) in enumerate(matches, start=1)
            ]
            correct = [result for result in ranked if result["chunk_id"] in expected_ids]
            best_correct = min(correct, key=lambda result: result["rank"]) if correct else None
            rows.append(
                {
                    "query_id": item["id"],
                    "query": item["query"],
                    "expected_chunk_ids": item["expected_chunk_ids"],
                    "correct_rank": best_correct["rank"] if best_correct else None,
                    "correct_score": best_correct["score"] if best_correct else None,
                    "correct_chunks": correct,
                    "top_20": ranked[:20],
                    "top_5_incorrect": [
                        result for result in ranked[:5] if result["chunk_id"] not in expected_ids
                    ],
                    "latency_ms": latency_ms,
                    "main_failure_reason": None,
                    "recommended_next_action": None,
                }
            )

    return {
        "dataset": dataset["name"],
        "knowledge_base": dataset["knowledge_base"],
        "embedding_model": MODEL_NAME,
        "corpus_size": corpus_size,
        "queries": rows,
    }


async def main() -> None:
    try:
        payload = await run()
        notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        for row in payload["queries"]:
            row.update(notes[row["query_id"]])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = OUTPUT_DIR / "failure_analysis.json"
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown = [
            "# Top-20 Retrieval Failure Analysis",
            "",
            f"Corpus size: {payload['corpus_size']} chunks",
            f"Embedding model: `{payload['embedding_model']}`",
        ]
        for row in payload["queries"]:
            markdown.extend(
                [
                    "",
                    f"## {row['query_id']} — {row['query']}",
                    "",
                    f"- Expected chunk IDs: `{row['expected_chunk_ids']}`",
                    f"- Correct rank: {row['correct_rank']}",
                    f"- Correct score: {row['correct_score']:.4f}",
                    f"- Diagnostic latency: {row['latency_ms']:.2f} ms",
                    f"- Main failure reason: {row['main_failure_reason']}",
                    f"- Recommended next action: {row['recommended_next_action']}",
                    "",
                    "| Rank | Chunk ID | Score | File | Pages | Section |",
                    "|---:|---|---:|---|---|---|",
                ]
            )
            for result in row["top_20"]:
                pages = ", ".join(str(page) for page in result["page_numbers"])
                section = " > ".join(result["section_path"]).replace("|", "\\|")
                markdown.append(
                    f"| {result['rank']} | `{result['chunk_id']}` | {result['score']:.4f} | "
                    f"{result['source_file']} | {pages} | {section} |"
                )
        (OUTPUT_DIR / "failure_analysis.md").write_text("\n".join(markdown), encoding="utf-8")
        print(
            json.dumps(
                [
                    {
                        "query_id": row["query_id"],
                        "correct_rank": row["correct_rank"],
                        "correct_score": row["correct_score"],
                        "latency_ms": row["latency_ms"],
                    }
                    for row in payload["queries"]
                ],
                indent=2,
            )
        )
    finally:
        await close_database_connection()


if __name__ == "__main__":
    asyncio.run(main())
