from sqlmodel import SQLModel, create_engine, Session
from app.core.config import settings

# SQLite engine (zero setup — file-based persistence)
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}  # needed for SQLite
)


def init_db():
    """Create all tables on startup."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency for DB sessions."""
    with Session(engine) as session:
        yield session
