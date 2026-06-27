from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Studio"
    VERSION: str = "1.0.0"
    
    # Database — defaults to SQLite (zero setup)
    DATABASE_URL: str = "sqlite:///./rag_studio.db"
    
    # AI Provider API Keys (optional — app works in DEMO_MODE without them)
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Demo Mode: if True, use simulated responses (no API keys required)
    DEMO_MODE: bool = True
    
    # Qdrant — uses in-memory mode by default (no Docker required)
    QDRANT_HOST: Optional[str] = None   # e.g. "localhost"
    QDRANT_PORT: int = 6333
    
    # Embedding dimensions for default provider
    EMBEDDING_DIMENSIONS: int = 768
    
    # LLM defaults
    DEFAULT_LLM_MODEL: str = "gemini/gemini-2.0-flash"
    DEFAULT_EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # File storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
