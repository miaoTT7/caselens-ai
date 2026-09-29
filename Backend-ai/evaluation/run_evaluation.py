"""Ingest local evaluation PDFs and run the pgvector retrieval baseline."""

from __future__ import annotations

import argparse
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
from api.ingestion import DocumentIngestionService  # noqa: E402
from api.repositories import ChunkRepository, DocumentRepository, KnowledgeBaseRepository  # noqa: E402
from api.retrieval import SemanticRetrievalService  # noqa: E402
from api.schemas import SemanticSearchRequest  # noqa: E402


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EVALUATION_DIR / "dataset.json"
DEFAULT_PDF_DIR = EVALUATION_DIR / "pdfs"
DEFAULT_RESULTS = EVALUATION_DIR / "results"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--skip-ingestion", action="store_true")
    return parser.parse_args()


def rank_for(query: dict, results: list[dict]) -> int | None:
    expected_chunk_ids = set(query.get("expected_chunk_ids", []))
    if not expected_chunk_ids:
        return None
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in expected_chunk_ids:
            return rank
    return 0


def calculate_metrics(rows: list[dict]) -> dict | None:
    evaluated = [row for row in rows if row["correct_rank"] is not None]
    ranks = [row["correct_rank"] for row in evaluated]
    if not ranks:
        return None
    total = len(ranks)
    return {
        "evaluated_queries": total,
        "hit_at_1": sum(row["hit_at_1"] for row in evaluated) / total,
        "hit_at_3": sum(row["hit_at_3"] for row in evaluated) / total,
        "hit_at_5": sum(row["hit_at_5"] for row in evaluated) / total,
        "mrr": sum(row["rr"] for row in evaluated) / total,
        "average_latency_ms": sum(row["latency_ms"] for row in evaluated) / total,
    }


async def ensure_documents(session, model, knowledge_base_id, pdf_dir: Path) -> list[dict]:
    repository = DocumentRepository(session)
    chunks = ChunkRepository(session)
    existing = {
        document.name: document
        for document in await repository.list_for_knowledge_base(knowledge_base_id)
        if document.status == "completed"
    }
    documents = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        document = existing.get(path.name)
        if document is None:
            document = await DocumentIngestionService(session, model).ingest(
                knowledge_base_id=knowledge_base_id,
                file_path=path,
                original_filename=path.name,
            )
        stored_chunks = await chunks.list_for_document(document.id)
        documents.append(
            {
                "id": str(document.id),
                "name": document.name,
                "status": document.status,
                "chunk_count": len(stored_chunks),
            }
        )
    return documents


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# pgvector Retrieval Candidate Report",
        "",
        f"Knowledge base: `{payload['knowledge_base']}`",
        f"Embedding model: `{payload['embedding_model']}`",
        "",
        "## Documents",
        "",
    ]
    for document in payload["documents"]:
        lines.append(f"- `{document['name']}`: {document['chunk_count']} chunks, {document['status']}")
    if payload["metrics"]:
        metrics = payload["metrics"]
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Hit@1: {metrics['hit_at_1']:.4f}",
                f"- Hit@3: {metrics['hit_at_3']:.4f}",
                f"- Hit@5: {metrics['hit_at_5']:.4f}",
                f"- MRR: {metrics['mrr']:.4f}",
                f"- Average latency: {metrics['average_latency_ms']:.2f} ms",
            ]
        )
    for row in payload["queries"]:
        lines.extend(
            [
                "",
                f"## {row['id']} — {row['query']}",
                "",
                f"Expected chunk IDs: `{row['expected_chunk_ids']}`  ",
                f"Top-5 chunk IDs: `{row['top_5_chunk_ids']}`  ",
                f"Correct rank: {row['correct_rank']}  ",
                f"Hit@1 / Hit@3 / Hit@5: {row['hit_at_1']} / {row['hit_at_3']} / {row['hit_at_5']}  ",
                f"RR: {row['rr']:.4f}  ",
                f"Latency: {row['latency_ms']:.2f} ms",
                "",
            ]
        )
        for rank, result in enumerate(row["results"], start=1):
            pages = ", ".join(str(page) for page in result["page_numbers"]) or "n/a"
            section = " > ".join(result["section_path"]) or "n/a"
            excerpt = " ".join(result["text"].split())[:500]
            lines.extend(
                [
                    f"### {rank}. {result['source_file']} — score {result['score']:.4f}",
                    "",
                    f"Pages: {pages}  ",
                    f"Section: {section}",
                    "",
                    excerpt,
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


async def run(args: argparse.Namespace) -> None:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    model = SentenceTransformer(EMBEDDING_MODEL)
    session_factory = get_session_factory()
    async with session_factory() as session:
        knowledge_bases = KnowledgeBaseRepository(session)
        knowledge_base = await knowledge_bases.get_by_name(dataset["knowledge_base"])
        if knowledge_base is None:
            knowledge_base = await knowledge_bases.create(
                name=dataset["knowledge_base"],
                description="Retrieval evaluation documents",
            )
            await session.commit()

        if args.skip_ingestion:
            chunks = ChunkRepository(session)
            documents = []
            for item in await DocumentRepository(session).list_for_knowledge_base(knowledge_base.id):
                stored_chunks = await chunks.list_for_document(item.id)
                documents.append(
                    {
                        "id": str(item.id),
                        "name": item.name,
                        "status": item.status,
                        "chunk_count": len(stored_chunks),
                    }
                )
        else:
            documents = await ensure_documents(session, model, knowledge_base.id, args.pdf_dir)

        retrieval = SemanticRetrievalService(session, model)
        rows = []
        for query in dataset["queries"]:
            started = perf_counter()
            matches = await retrieval.search(
                knowledge_base.id,
                SemanticSearchRequest(query=query["query"], limit=args.limit),
            )
            latency_ms = (perf_counter() - started) * 1000
            results = [result.model_dump(mode="json") for result in matches]
            correct_rank = rank_for(query, results)
            rows.append(
                {
                    "id": query["id"],
                    "category": query["category"],
                    "query": query["query"],
                    "expected_chunk_ids": query.get("expected_chunk_ids", []),
                    "top_5_chunk_ids": [result["chunk_id"] for result in results[:5]],
                    "correct_rank": correct_rank,
                    "hit_at_1": correct_rank == 1 if correct_rank is not None else None,
                    "hit_at_3": 0 < correct_rank <= 3 if correct_rank is not None else None,
                    "hit_at_5": 0 < correct_rank <= 5 if correct_rank is not None else None,
                    "rr": 1 / correct_rank if correct_rank and correct_rank > 0 else 0.0,
                    "latency_ms": latency_ms,
                    "results": results,
                }
            )

    payload = {
        "dataset": dataset["name"],
        "knowledge_base": dataset["knowledge_base"],
        "embedding_model": EMBEDDING_MODEL,
        "top_k": args.limit,
        "documents": documents,
        "metrics": calculate_metrics(rows),
        "queries": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(args.output_dir / "candidate_report.md", payload)
    print(json.dumps({"documents": documents, "metrics": payload["metrics"]}, indent=2))


async def main() -> None:
    try:
        await run(parse_args())
    finally:
        await close_database_connection()


if __name__ == "__main__":
    asyncio.run(main())
