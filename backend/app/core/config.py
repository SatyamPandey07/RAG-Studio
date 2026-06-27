import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Studio"
    API_V1_STR: str = "/api/v1"
    
    # Databases
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./rag_studio.db")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # API Keys (Loaded from env or .env)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    VOYAGE_API_KEY: Optional[str] = os.getenv("VOYAGE_API_KEY")
    JINA_API_KEY: Optional[str] = os.getenv("JINA_API_KEY")
    
    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()
