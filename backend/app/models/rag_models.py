from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, Dict
from datetime import datetime
import json

class Workspace(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    projects: List["Project"] = Relationship(back_populates="workspace")

class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    workspace_id: int = Field(foreign_key="workspace.id")
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    workspace: Workspace = Relationship(back_populates="projects")
    collections: List["Collection"] = Relationship(back_populates="project")
    pipelines: List["Pipeline"] = Relationship(back_populates="project")

class Collection(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    project: Project = Relationship(back_populates="collections")
    documents: List["Document"] = Relationship(back_populates="collection")

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    collection_id: int = Field(foreign_key="collection.id")
    name: str
    file_type: str
    size_bytes: int
    path: str
    content_hash: str
    status: str = "Ingested" # Ingested, Processing, Failed, Completed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    collection: Collection = Relationship(back_populates="documents")
    chunks: List["Chunk"] = Relationship(back_populates="document")

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id")
    index: int
    text_content: str
    token_count: int
    metadata_json: str = "{}"
    
    document: Document = Relationship(back_populates="chunks")

class Pipeline(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id")
    name: str
    
    # Store settings as JSON strings
    chunking_config: str = "{}" # size, overlap, strategy
    embedding_config: str = "{}" # provider, model, dimensions
    db_config: str = "{}" # host, collection, metric
    retriever_config: str = "{}" # hybrid, dense_weight, sparse_weight, rerank
    llm_config: str = "{}" # provider, model, temperature, prompt
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    project: Project = Relationship(back_populates="pipelines")

class EvaluationRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_id: int = Field(foreign_key="pipeline.id")
    name: str
    status: str = "Pending" # Pending, Running, Completed, Failed
    
    # Scores (0.0 to 1.0)
    recall: float = 0.0
    precision: float = 0.0
    faithfulness: float = 0.0
    groundedness: float = 0.0
    
    # Incurred costs
    cost: float = 0.0
    latency_ms: int = 0
    token_count: int = 0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class AnalyticsLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_id: int = Field(foreign_key="pipeline.id")
    query: str
    response: str
    latency_ms: int
    cost_usd: float
    feedback_score: Optional[int] = None # +1 (thumbs up), -1 (thumbs down)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
