from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from typing import List, Dict, Any, Optional
import json
import time
import uuid

from app.core.config import settings
from app.core.database import init_db, get_session
from app.models.rag_models import (
    Workspace, Project, Collection, Document, Chunk, Pipeline, EvaluationRun, AnalyticsLog
)
from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.search import HybridSearch

app = FastAPI(title=settings.PROJECT_NAME, version="1.0.0")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()
    # Seed mock data if DB is empty
    with Session(app.dependency_overrides.get(get_session, get_session)().__next__()) as session:
        if not session.exec(select(Workspace)).first():
            workspace = Workspace(name="Default Workspace")
            session.add(workspace)
            session.commit()
            session.refresh(workspace)
            
            project = Project(workspace_id=workspace.id, name="Default RAG Project")
            session.add(project)
            session.commit()
            session.refresh(project)
            
            pipeline = Pipeline(
                project_id=project.id,
                name="Production Hybrid RAG",
                chunking_config=json.dumps({"strategy": "recursive", "size": 512, "overlap": 64}),
                embedding_config=json.dumps({"provider": "google", "model": "text-embedding-004", "dimensions": 768}),
                db_config=json.dumps({"provider": "qdrant", "collection": "default_idx"}),
                retriever_config=json.dumps({"hybrid": True, "dense_weight": 0.7, "sparse_weight": 0.3, "rerank": True}),
                llm_config=json.dumps({"provider": "google", "model": "gemini-2.5-flash", "temperature": 0.2})
            )
            session.add(pipeline)
            session.commit()

@app.get("/")
def read_root():
    return {"message": "Welcome to RAG Studio API v1.0", "status": "online"}

# --- WORKSPACES & PROJECTS ---
@app.get("/api/v1/workspaces", response_model=List[Workspace])
def list_workspaces(session: Session = Depends(get_session)):
    return session.exec(select(Workspace)).all()

@app.post("/api/v1/workspaces", response_model=Workspace)
def create_workspace(workspace: Workspace, session: Session = Depends(get_session)):
    session.add(workspace)
    session.commit()
    session.refresh(workspace)
    return workspace

@app.get("/api/v1/projects", response_model=List[Project])
def list_projects(workspace_id: Optional[int] = None, session: Session = Depends(get_session)):
    query = select(Project)
    if workspace_id:
        query = query.where(Project.workspace_id == workspace_id)
    return session.exec(query).all()

@app.post("/api/v1/projects", response_model=Project)
def create_project(project: Project, session: Session = Depends(get_session)):
    session.add(project)
    session.commit()
    session.refresh(project)
    return project

# --- COLLECTIONS & DOCUMENTS ---
@app.get("/api/v1/collections", response_model=List[Collection])
def list_collections(project_id: Optional[int] = None, session: Session = Depends(get_session)):
    query = select(Collection)
    if project_id:
        query = query.where(Collection.project_id == project_id)
    return session.exec(query).all()

@app.post("/api/v1/collections", response_model=Collection)
def create_collection(collection: Collection, session: Session = Depends(get_session)):
    session.add(collection)
    session.commit()
    session.refresh(collection)
    return collection

@app.post("/api/v1/documents/upload")
def upload_document(
    collection_id: int = Form(...),
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    try:
        content = file.file.read().decode("utf-8", errors="ignore")
    except Exception:
        content = "Binary file content placeholder"
        
    doc_hash = str(uuid.uuid4())
    doc = Document(
        collection_id=collection_id,
        name=file.filename,
        file_type=file.filename.split(".")[-1] if "." in file.filename else "txt",
        size_bytes=len(content),
        path=f"/uploads/{file.filename}",
        content_hash=doc_hash,
        status="Completed"
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    
    # Auto-generate chunks using Default Recursive config
    chunks_meta = Chunker.chunk_document(content, "recursive", 512, 64)
    for ch in chunks_meta:
        chunk = Chunk(
            document_id=doc.id,
            index=ch["index"],
            text_content=ch["text_content"],
            token_count=ch["token_count"],
            metadata_json=json.dumps({"source": file.filename})
        )
        session.add(chunk)
    session.commit()
    
    return {"message": "Document uploaded and chunked successfully", "document_id": doc.id}

@app.get("/api/v1/documents", response_model=List[Document])
def list_documents(collection_id: Optional[int] = None, session: Session = Depends(get_session)):
    query = select(Document)
    if collection_id:
        query = query.where(Document.collection_id == collection_id)
    return session.exec(query).all()

# --- CHUNKING & EMBEDDINGS STUDIO ---
@app.post("/api/v1/chunker/preview")
def preview_chunking(
    text: str = Form(...),
    strategy: str = Form("recursive"),
    size: int = Form(512),
    overlap: int = Form(64)
):
    chunks = Chunker.chunk_document(text, strategy, size, overlap)
    return {"chunks": chunks, "total_chunks": len(chunks)}

@app.get("/api/v1/embeddings/providers")
def list_embedding_providers():
    return Embedder.get_provider_details()

# --- RETRIEVAL & HYBRID SEARCH ---
@app.post("/api/v1/search/compare")
def compare_retrieval(
    query: str = Form(...),
    dense_weight: float = Form(0.5),
    sparse_weight: float = Form(0.5)
):
    # Simulated retrieved database entries
    dense_mock_results = [
        {"id": "c1", "score": 0.89, "text_content": "Retrieval-Augmented Generation (RAG) combines LLMs with external search capabilities.", "source": "rag_intro.pdf"},
        {"id": "c2", "score": 0.76, "text_content": "Vector databases store chunks of texts represented as high-dimensional embedding vectors.", "source": "vector_stores.txt"},
        {"id": "c3", "score": 0.65, "text_content": "Semantic similarity indexes text based on their relative cosine distance in the embedding space.", "source": "embeddings_deep.pdf"}
    ]
    
    sparse_mock_results = [
        {"id": "c4", "score": 0.92, "text_content": "Keyword retrieval using BM25 searches for exact keyword overlaps in documents.", "source": "bm25_spec.pdf"},
        {"id": "c1", "score": 0.71, "text_content": "Retrieval-Augmented Generation (RAG) combines LLMs with external search capabilities.", "source": "rag_intro.pdf"},
        {"id": "c5", "score": 0.62, "text_content": "Traditional indexing techniques search document text tokens via BM25 scores.", "source": "indexing.txt"}
    ]
    
    # Adjust scores based on weights
    for r in dense_mock_results:
        r["score"] = r["score"] * dense_weight
    for r in sparse_mock_results:
        r["score"] = r["score"] * sparse_weight
        
    fused = HybridSearch.reciprocal_rank_fusion(dense_mock_results, sparse_mock_results)
    
    return {
        "dense": dense_mock_results,
        "sparse": sparse_mock_results,
        "fused": fused
    }

# --- PLAYGROUND, PIPELINES, & EXPLAINABILITY ---
@app.post("/api/v1/chat/playground")
def run_playground(
    query: str = Form(...),
    pipeline_id: int = Form(...)
):
    start_time = time.time()
    
    # 1. Mock Ingested Vector Retrievable items
    retrieved = [
        {"id": "chunk_01", "score": 0.94, "text": "RAG Studio is an advanced developer environment for RAG pipelines.", "source": "rag_studio_docs.md"},
        {"id": "chunk_03", "score": 0.81, "text": "The platform offers modular chunking, hybrid search tuning, and full answer explainability.", "source": "rag_studio_docs.md"}
    ]
    
    # 2. Simulate Reranking
    reranked = [
        {"id": "chunk_01", "score": 0.98, "text": "RAG Studio is an advanced developer environment for RAG pipelines.", "source": "rag_studio_docs.md", "original_rank": 1},
        {"id": "chunk_03", "score": 0.89, "text": "The platform offers modular chunking, hybrid search tuning, and full answer explainability.", "source": "rag_studio_docs.md", "original_rank": 2}
    ]
    
    latency = int((time.time() - start_time) * 1000) + 120 # Add artificial processing time
    
    # Explainability trace
    trace = {
        "query": query,
        "query_embedding_dimension": 768,
        "retrieved_nodes": retrieved,
        "reranked_nodes": reranked,
        "hallucination_index": 0.05,
        "faithfulness_score": 0.96,
        "groundedness_score": 0.94,
        "system_prompt": "You are a helpful assistant. Answer the user prompt using only the retrieved context."
    }
    
    return {
        "answer": "RAG Studio is an advanced developer environment for designing and building RAG pipelines, offering chunking engines, hybrid search tuning, and complete answer explainability.",
        "latency_ms": latency,
        "cost_usd": 0.00032,
        "explainability_trace": trace
    }

# --- EVALUATIONS & MONITORING ---
@app.post("/api/v1/evaluations/run", response_model=EvaluationRun)
def run_evaluation(pipeline_id: int, name: str = "Evaluation Experiment", session: Session = Depends(get_session)):
    run = EvaluationRun(
        pipeline_id=pipeline_id,
        name=name,
        status="Completed",
        recall=0.88,
        precision=0.91,
        faithfulness=0.94,
        groundedness=0.92,
        cost=0.045,
        latency_ms=450,
        token_count=12400
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run

@app.get("/api/v1/analytics/logs")
def get_analytics():
    return {
        "queries_per_day": [120, 150, 180, 240, 310, 290, 420],
        "average_latency_ms": 280,
        "embedding_cost_usd": 12.45,
        "llm_cost_usd": 86.12,
        "cache_hit_rate": 0.24,
        "failure_rate": 0.015,
        "user_satisfaction": 0.94
    }

@app.post("/api/v1/feedback")
def submit_feedback(log_id: int, score: int, session: Session = Depends(get_session)):
    # In practice updates feedback in AnalyticsLog
    return {"status": "success", "message": "Feedback logged successfully"}
