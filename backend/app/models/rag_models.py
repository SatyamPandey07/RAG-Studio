from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# Workspace / Project / Collection hierarchy
# ─────────────────────────────────────────────────────────────

class Workspace(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    projects: List["Project"] = Relationship(back_populates="workspace")


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    workspace_id: int = Field(foreign_key="workspace.id")
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    workspace: Optional[Workspace] = Relationship(back_populates="projects")
    collections: List["Collection"] = Relationship(back_populates="project")
    pipelines: List["Pipeline"] = Relationship(back_populates="project")


class Collection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    description: Optional[str] = None
    embedding_model: str = "text-embedding-004"
    vector_dimensions: int = 768
    created_at: datetime = Field(default_factory=datetime.utcnow)

    project: Optional[Project] = Relationship(back_populates="collections")
    documents: List["Document"] = Relationship(back_populates="collection")


# ─────────────────────────────────────────────────────────────
# Documents & Chunks
# ─────────────────────────────────────────────────────────────

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collection.id")
    name: str
    file_type: str
    size_bytes: int = 0
    path: str = ""
    content_hash: str = ""
    status: str = "Processing"   # Processing | Completed | Failed
    chunk_count: int = 0
    word_count: int = 0
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    collection: Optional[Collection] = Relationship(back_populates="documents")
    chunks: List["Chunk"] = Relationship(back_populates="document")


class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    collection_id: int = Field(default=0)
    index: int
    text_content: str
    token_count: int = 0
    char_count: int = 0
    metadata_json: str = "{}"
    embedding_stored: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Optional[Document] = Relationship(back_populates="chunks")


# ─────────────────────────────────────────────────────────────
# Pipelines & Prompt Templates
# ─────────────────────────────────────────────────────────────

class Pipeline(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    description: Optional[str] = None
    is_active: bool = True

    # Configuration stored as JSON strings
    chunking_config: str = '{}'   # strategy, size, overlap
    embedding_config: str = '{}'  # provider, model, dimensions
    db_config: str = '{}'         # provider, collection
    retriever_config: str = '{}'  # top_k, dense_weight, sparse_weight, mmr
    rerank_config: str = '{}'     # enabled, model, top_n
    llm_config: str = '{}'        # provider, model, temperature, max_tokens
    system_prompt: str = "You are a helpful assistant. Answer the user's question using ONLY the retrieved context. If the context doesn't contain the answer, say 'I don't have enough information to answer this question.'"

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    project: Optional[Project] = Relationship(back_populates="pipelines")
    eval_runs: List["EvaluationRun"] = Relationship(back_populates="pipeline")
    chat_sessions: List["ChatSession"] = Relationship(back_populates="pipeline")


class PromptTemplate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    system_prompt: str = ""
    user_prompt_template: str = "Context:\n{context}\n\nQuestion: {question}"
    version: int = 1
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# Chat Sessions & Messages
# ─────────────────────────────────────────────────────────────

class ChatSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_id: int = Field(foreign_key="pipeline.id")
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    pipeline: Optional[Pipeline] = Relationship(back_populates="chat_sessions")
    messages: List["ChatMessage"] = Relationship(back_populates="session")


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="chatsession.id")
    role: str  # user | assistant
    content: str
    # For assistant messages — stored as JSON
    retrieved_chunks_json: Optional[str] = None   # list of {text, source, score}
    explainability_trace_json: Optional[str] = None
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    token_count: Optional[int] = None
    hallucination_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    feedback: Optional[int] = None   # +1 thumbs up, -1 thumbs down
    created_at: datetime = Field(default_factory=datetime.utcnow)

    session: Optional[ChatSession] = Relationship(back_populates="messages")


# ─────────────────────────────────────────────────────────────
# Evaluation Runs
# ─────────────────────────────────────────────────────────────

class EvaluationRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_id: int = Field(foreign_key="pipeline.id")
    name: str
    status: str = "Pending"   # Pending | Running | Completed | Failed

    # Retrieval metrics
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    # Generation quality metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    groundedness: float = 0.0
    hallucination_rate: float = 0.0
    citation_accuracy: float = 0.0

    # Cost & performance
    avg_latency_ms: int = 0
    avg_cost_usd: float = 0.0
    total_tokens: int = 0
    queries_evaluated: int = 0

    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    pipeline: Optional[Pipeline] = Relationship(back_populates="eval_runs")


# ─────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────

class AnalyticsLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_id: Optional[int] = Field(default=None, foreign_key="pipeline.id")
    session_id: Optional[int] = None
    query: str
    response: str
    latency_ms: int = 0
    cost_usd: float = 0.0
    token_count: int = 0
    feedback_score: Optional[int] = None
    hallucination_score: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────────────────────

class APIKey(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    key_prefix: str   # e.g. "rs_live_xxxx" (we only store prefix for display)
    key_hash: str     # SHA-256 of the full key
    is_active: bool = True
    rate_limit_rpm: int = 60
    total_requests: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
