"""
Vector Store Service — Qdrant in-memory singleton.

In DEMO_MODE or when no Qdrant host is configured, uses qdrant_client's
in-process InMemoryClient. Supports upsert, dense cosine search, and deletion.
"""
from typing import List, Dict, Any, Optional
import uuid


class VectorStore:
    _client = None
    _collections: set = set()

    @classmethod
    def _get_client(cls):
        if cls._client is None:
            try:
                from qdrant_client import QdrantClient
                from app.core.config import settings
                if settings.QDRANT_HOST:
                    cls._client = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT
                    )
                else:
                    # In-memory mode — no Docker required
                    cls._client = QdrantClient(":memory:")
            except Exception as e:
                print(f"[VectorStore] Failed to init Qdrant: {e}")
                cls._client = None
        return cls._client

    @classmethod
    def _ensure_collection(cls, collection_name: str, vector_size: int = 768):
        client = cls._get_client()
        if client is None:
            return
        if collection_name not in cls._collections:
            try:
                from qdrant_client.models import Distance, VectorParams
                existing = [c.name for c in client.get_collections().collections]
                if collection_name not in existing:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                cls._collections.add(collection_name)
            except Exception as e:
                print(f"[VectorStore] Error creating collection: {e}")

    @classmethod
    def upsert_chunks(
        cls,
        collection_name: str,
        chunks: List[Dict[str, Any]],  # [{id, vector, text, metadata}]
        vector_size: int = 768
    ) -> bool:
        """Upsert chunk vectors into Qdrant."""
        client = cls._get_client()
        if client is None or not chunks:
            return False
        cls._ensure_collection(collection_name, vector_size)
        try:
            from qdrant_client.models import PointStruct
            points = [
                PointStruct(
                    id=chunk["id"],
                    vector=chunk["vector"],
                    payload={
                        "text": chunk["text"],
                        "source": chunk.get("source", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "document_id": chunk.get("document_id", 0),
                        "collection_id": chunk.get("collection_id", 0),
                    }
                )
                for chunk in chunks
            ]
            client.upsert(collection_name=collection_name, points=points)
            return True
        except Exception as e:
            print(f"[VectorStore] Upsert error: {e}")
            return False

    @classmethod
    def search(
        cls,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Dense cosine similarity search."""
        client = cls._get_client()
        if client is None:
            return []
        cls._ensure_collection(collection_name, len(query_vector))
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            return [
                {
                    "id": str(r.id),
                    "score": round(r.score, 4),
                    "text": r.payload.get("text", ""),
                    "source": r.payload.get("source", ""),
                    "chunk_index": r.payload.get("chunk_index", 0),
                    "document_id": r.payload.get("document_id", 0),
                }
                for r in results
            ]
        except Exception as e:
            print(f"[VectorStore] Search error: {e}")
            return []

    @classmethod
    def get_collection_info(cls, collection_name: str) -> Dict[str, Any]:
        client = cls._get_client()
        if client is None:
            return {"vectors_count": 0, "status": "disconnected"}
        try:
            info = client.get_collection(collection_name)
            return {
                "vectors_count": info.vectors_count or 0,
                "status": str(info.status),
                "config": {
                    "size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                }
            }
        except Exception:
            return {"vectors_count": 0, "status": "empty"}

    @classmethod
    def collection_name_for(cls, collection_id: int) -> str:
        return f"collection_{collection_id}"
