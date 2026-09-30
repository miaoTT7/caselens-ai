from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, UploadFile, Form, HTTPException, status
from dotenv import load_dotenv
import os
import tempfile
import uuid
import json
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from api.database import check_database_connection, close_database_connection, get_database_session
from api.document_chunker import chunk_document
from api.document_parser import parse_document, parse_docx, parse_pdf
from api.claim_fact_extraction import ClaimFactExtractionService
from api.claim_fact_validation import ClaimFactValidationService
from api.coverage_assessment import CoverageAssessmentService
from api.exclusion_assessment import ExclusionAssessmentService
from api.obligation_assessment import ObligationAssessmentService
from api.claim_calculation import ClaimCalculationService
from api.claim_recommendation import ClaimRecommendationService
from api.claim_assessment import ClaimAssessmentOrchestrator
from api.claim_schemas import (
    ClaimFactExtractionRequest,
    ClaimFactExtractionResponse,
    FactValidationRequest,
    FactValidationResponse,
    CoverageAssessmentRequest,
    CoverageAssessmentResponse,
    ExclusionAssessmentRequest,
    ExclusionAssessmentResponse,
    ObligationAssessmentRequest,
    ObligationAssessmentResponse,
    ClaimCalculationRequest,
    ClaimCalculationResponse,
    ClaimRecommendationRequest,
    ClaimRecommendationResponse,
    ClaimAssessmentRequest,
    ClaimAssessmentResponse,
)
from api.ingestion import DocumentIngestionService
from api.llm_service import GroundedAnswerService, LLMProviderError
from api.repositories import ChunkRepository, DocumentRepository, KnowledgeBaseRepository
from api.retrieval import SemanticRetrievalService
from api.schemas import (
    ChunkResponse,
    DocumentResponse,
    GroundedAskRequest,
    GroundedAskResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


# ---------- Setup ----------


BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(BACKEND_ROOT, ".env"), override=True)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


nlp = None


def _legacy_nlp():
    """Load spaCy only when a legacy FAISS upload route needs it."""
    global nlp
    if nlp is None:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return None
    return nlp

model = SentenceTransformer("all-MiniLM-L6-v2")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await check_database_connection()
    yield
    await close_database_connection()


app = FastAPI(lifespan=lifespan)
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ---------- Utilities ----------

def parse_and_enhance_query(user_query):
    pipeline = _legacy_nlp()
    if pipeline is None:
        return user_query
    doc = pipeline(user_query)
    keywords = []
    
    # Extract proper nouns, nouns, medical terms, numbers, locations, dates
    for token in doc:
        if token.ent_type_ in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
            keywords.append(token.text)
        elif token.ent_type_ in ['GPE', 'LOC']:
            keywords.append(token.text)
        elif token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
            keywords.append(token.lemma_)  # Lemmatize to improve match
    
    # Join enhanced query
    enhanced_query = " ".join(keywords)
    
    return enhanced_query if enhanced_query else user_query



def process_claim(user_query: str, clause: str):
    import requests

    prompt = f"""
You are an insurance claim analyst. Based on the user query and clause, decide the claim outcome.

Query: {user_query}

Clause: {clause}

Analyze if the claim should be approved or rejected based on the clause conditions. Return only a JSON response with these exact fields:
- decision: "Approved" or "Rejected"
- amount: estimated amount like "₹50000" or "N/A" if rejected
- justification: brief explanation based on the clause

Response:
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ],
                "temperature": 0.3
            }
        )

        result = response.json()
        print(result)

        if "choices" not in result or not result["choices"]:
            return {"error": "Unexpected response from Groq", "raw": result}

        content = result["choices"][0]["message"]["content"]
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            return json.loads(content[json_start:json_end])
        return {"error": "Could not parse JSON", "raw": content}

    except Exception as e:
        return {"error": str(e)}
    
    
# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Insurance Claim API running with Groq API for LLM."}


@app.get("/health/database")
async def database_health():
    try:
        return await check_database_connection()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {error}") from error


@app.post(
    "/knowledge-bases",
    response_model=KnowledgeBaseResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    session: AsyncSession = Depends(get_database_session),
):
    repository = KnowledgeBaseRepository(session)
    try:
        knowledge_base = await repository.create(
            name=payload.name.strip(), description=payload.description
        )
        await session.commit()
        return knowledge_base
    except IntegrityError as error:
        await session.rollback()
        raise HTTPException(status_code=409, detail="A knowledge base with this name already exists") from error


@app.get("/knowledge-bases", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(session: AsyncSession = Depends(get_database_session)):
    return await KnowledgeBaseRepository(session).list()


@app.get("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    knowledge_base_id: uuid.UUID,
    session: AsyncSession = Depends(get_database_session),
):
    knowledge_base = await KnowledgeBaseRepository(session).get(knowledge_base_id)
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return knowledge_base


@app.post(
    "/knowledge-bases/{knowledge_base_id}/search",
    response_model=SemanticSearchResponse,
)
async def search_knowledge_base(
    knowledge_base_id: uuid.UUID,
    payload: SemanticSearchRequest,
    session: AsyncSession = Depends(get_database_session),
):
    try:
        results = await SemanticRetrievalService(session, model).search(knowledge_base_id, payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return SemanticSearchResponse(query=payload.query.strip(), results=results)


@app.post("/ask", response_model=GroundedAskResponse)
async def ask_question(
    payload: GroundedAskRequest,
    session: AsyncSession = Depends(get_database_session),
):
    query = payload.query.strip()
    try:
        results = await SemanticRetrievalService(session, model).search(
            payload.knowledge_base_id,
            SemanticSearchRequest(
                query=query,
                limit=payload.limit,
                retrieval_mode="hybrid_rerank",
            ),
        )
        answer, citations = await GroundedAnswerService().answer(query, results)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Grounded answer generation failed: {error}") from error
    return GroundedAskResponse(
        query=query,
        answer=answer.answer,
        evidence_ids=answer.evidence_ids,
        insufficient_evidence=answer.insufficient_evidence,
        citations=citations,
    )


@app.post("/claim-facts/extract", response_model=ClaimFactExtractionResponse)
async def extract_claim_facts(payload: ClaimFactExtractionRequest):
    try:
        return await ClaimFactExtractionService().extract(payload)
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Claim fact extraction failed: {error}") from error


@app.post("/claim-facts/validate", response_model=FactValidationResponse)
async def validate_claim_facts(payload: FactValidationRequest):
    return ClaimFactValidationService().validate(payload)


@app.post("/coverage/assess", response_model=CoverageAssessmentResponse)
async def assess_coverage(
    payload: CoverageAssessmentRequest,
    session: AsyncSession = Depends(get_database_session),
):
    try:
        return await CoverageAssessmentService(session, model).assess(payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Coverage assessment failed: {error}") from error


@app.post("/exclusions/assess", response_model=ExclusionAssessmentResponse)
async def assess_exclusions(
    payload: ExclusionAssessmentRequest,
    session: AsyncSession = Depends(get_database_session),
):
    try:
        return await ExclusionAssessmentService(session, model).assess(payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Exclusion assessment failed: {error}") from error


@app.post("/obligations/assess", response_model=ObligationAssessmentResponse)
async def assess_obligations(
    payload: ObligationAssessmentRequest,
    session: AsyncSession = Depends(get_database_session),
):
    try:
        return await ObligationAssessmentService(session, model).assess(payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Obligation assessment failed: {error}") from error


@app.post("/claims/calculate", response_model=ClaimCalculationResponse)
async def calculate_claim(
    payload: ClaimCalculationRequest,
    session: AsyncSession = Depends(get_database_session),
):
    try:
        return await ClaimCalculationService(session, model).calculate(payload)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=502, detail=f"Claim calculation failed: {error}") from error


@app.post("/claims/recommend", response_model=ClaimRecommendationResponse)
async def recommend_claim(payload: ClaimRecommendationRequest):
    return ClaimRecommendationService().recommend(payload)


def get_claim_assessment_orchestrator(
    session: AsyncSession = Depends(get_database_session),
) -> ClaimAssessmentOrchestrator:
    return ClaimAssessmentOrchestrator(session, model)


@app.post("/claims/assess", response_model=ClaimAssessmentResponse)
async def assess_claim(
    payload: ClaimAssessmentRequest,
    orchestrator: ClaimAssessmentOrchestrator = Depends(get_claim_assessment_orchestrator),
):
    return await orchestrator.assess(payload)


@app.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    knowledge_base_id: uuid.UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_database_session),
):
    filename = os.path.basename(file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    suffix = os.path.splitext(filename)[1]
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as temporary:
            temporary_path = temporary.name
            while content := await file.read(1024 * 1024):
                temporary.write(content)
        service = DocumentIngestionService(session, model)
        return await service.ingest(
            knowledge_base_id=knowledge_base_id,
            file_path=temporary_path,
            original_filename=filename,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {error}") from error
    finally:
        await file.close()
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_persisted_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_database_session),
):
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@app.get("/documents/{document_id}/chunks", response_model=list[ChunkResponse])
async def list_persisted_chunks(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_database_session),
):
    if await DocumentRepository(session).get(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return await ChunkRepository(session).list_for_document(document_id)


@app.post("/parse-document")
async def parse_uploaded_document(file: UploadFile = File(...)):
    """Parse a supported file without running retrieval or claim analysis."""
    filename = os.path.basename(file.filename or "")
    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required.")
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as destination:
        destination.write(await file.read())
    try:
        document = await run_in_threadpool(parse_document, file_path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {error}") from error
    chunks = await run_in_threadpool(chunk_document, document)
    return {
        "parsed_document": document.to_dict(),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), user_query: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Step 1: Save uploaded PDF
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Step 2: Run heavy logic in a background thread to prevent blocking
    def process_pdf():
        import faiss
        import numpy as np

        document = parse_pdf(file_path)
        chunks = chunk_document(document)
        real_clauses = [chunk.text for chunk in chunks]
        if not real_clauses:
            raise HTTPException(status_code=400, detail="No valid clauses found in PDF.")

        # Create FAISS index and embeddings
        clause_embeddings = model.encode(real_clauses)
        clause_embeddings_np = np.array(clause_embeddings)
        dimension = clause_embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(clause_embeddings_np)

        # Search for relevant clauses
        parsed_query = parse_and_enhance_query(user_query)
        query_embedding = model.encode([parsed_query])
        distances, indices = index.search(np.array(query_embedding), min(5, len(real_clauses)))
        matched_clauses = [real_clauses[i] for i in indices[0]]

        top_clauses = ', '.join(matched_clauses[:5])
        llm_result = process_claim(user_query, top_clauses)

        return {
            "parsed_document": document.to_dict(),
            "chunks": [chunk.to_dict() for chunk in chunks],
            "matched_clauses": matched_clauses,
            "LLM_response": llm_result
        }

    # Run in a separate thread
    result = await run_in_threadpool(process_pdf)

    return {
        "message": "File uploaded and processed successfully.",
        "user_query": user_query,
        "parsed_document": result["parsed_document"],
        "chunks": result["chunks"],
        "matched_clauses": result["matched_clauses"],
        "LLM_response": result["LLM_response"]
    }


@app.post("/upload-docs")
async def upload_doc(file: UploadFile = File(...), user_query: str = Form(...)):
    import faiss
    import numpy as np

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Parse document blocks, preserving paragraph and table order.
    try:
        document = parse_docx(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to read Word document.")

    chunks = chunk_document(document)
    real_clauses = [chunk.text for chunk in chunks]

    if not real_clauses:
        raise HTTPException(status_code=400, detail="No valid clauses found in Word document.")

    # Encode clauses
    clause_embeddings = model.encode(real_clauses)
    clause_embeddings_np = np.array(clause_embeddings).astype("float32")
    dimension = clause_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(clause_embeddings_np)

    # Query embedding
    parsed_query = parse_and_enhance_query(user_query)
    print("Parsed Query:", parsed_query)
    query_embedding = model.encode([parsed_query])
    distances, indices = index.search(np.array(query_embedding), min(5, len(real_clauses)))

    # Matched clauses
    matched_clauses = [real_clauses[i] for i in indices[0]]
    top_clauses = ', '.join(matched_clauses[:5])

    # LLM processing
    result = process_claim(user_query, top_clauses)
    print(result)

    return {
        "message": "Word document uploaded and processed successfully.",
        "user_query": user_query,
        "parsed_document": document.to_dict(),
        "chunks": [chunk.to_dict() for chunk in chunks],
        "matched_clauses": matched_clauses,
        "LLM_response": result
    }
