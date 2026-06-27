"""
Embedding Service — generates vector embeddings for text.

In DEMO_MODE: returns deterministic pseudo-random vectors (consistent for same input).
With GOOGLE_API_KEY: uses Google text-embedding-004 (768 dimensions).
With OPENAI_API_KEY: uses text-embedding-3-small (1536 dimensions).
"""
import hashlib
import math
import random
from typing import List, Optional
from app.core.config import settings


PROVIDER_INFO = {
    "google": {
        "model": "models/text-embedding-004",
        "dimensions": 768,
        "mteb_score": 66.2,
        "cost_per_1k_tokens": 0.000001,
        "context_window": 2048,
        "speed": "Fast",
        "languages": "100+",
    },
    "openai": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "mteb_score": 64.5,
        "cost_per_1k_tokens": 0.00002,
        "context_window": 8191,
        "speed": "Fast",
        "languages": "50+",
    },
    "voyage": {
        "model": "voyage-2",
        "dimensions": 1024,
        "mteb_score": 68.1,
        "cost_per_1k_tokens": 0.00010,
        "context_window": 4096,
        "speed": "Medium",
        "languages": "30+",
    },
    "cohere": {
        "model": "embed-english-v3.0",
        "dimensions": 1024,
        "mteb_score": 64.8,
        "cost_per_1k_tokens": 0.00010,
        "context_window": 512,
        "speed": "Medium",
        "languages": "100+",
    },
    "jina": {
        "model": "jina-embeddings-v2-base-en",
        "dimensions": 768,
        "mteb_score": 60.4,
        "cost_per_1k_tokens": 0.00002,
        "context_window": 8192,
        "speed": "Fast",
        "languages": "10+",
    },
}


class EmbeddingService:

    @staticmethod
    def get_provider_details() -> dict:
        return PROVIDER_INFO

    @staticmethod
    def _demo_embed(text: str, dimensions: int = 768) -> List[float]:
        """
        Generate a deterministic pseudo-random vector from text.
        Same text always produces the same vector.
        Not semantically meaningful but useful for demo/testing.
        """
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2 ** 32)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(dimensions)]
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def embed(texts: List[str], provider: str = "google") -> List[List[float]]:
        """
        Embed a list of texts. Returns list of float vectors.
        """
        info = PROVIDER_INFO.get(provider, PROVIDER_INFO["google"])
        dims = info["dimensions"]

        # Demo mode — fast, no API calls
        if settings.DEMO_MODE or not settings.GOOGLE_API_KEY:
            return [EmbeddingService._demo_embed(t, dims) for t in texts]

        # Real Google embedding
        if provider == "google" and settings.GOOGLE_API_KEY:
            return EmbeddingService._google_embed(texts, dims)

        # Real OpenAI embedding
        if provider == "openai" and settings.OPENAI_API_KEY:
            return EmbeddingService._openai_embed(texts, dims)

        # Fallback to demo
        return [EmbeddingService._demo_embed(t, dims) for t in texts]

    @staticmethod
    def embed_query(query: str, provider: str = "google") -> List[float]:
        results = EmbeddingService.embed([query], provider)
        return results[0] if results else []

    @staticmethod
    def _google_embed(texts: List[str], dimensions: int) -> List[List[float]]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            results = []
            for text in texts:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                results.append(result["embedding"])
            return results
        except Exception as e:
            print(f"[EmbeddingService] Google embed error: {e}, falling back to demo")
            return [EmbeddingService._demo_embed(t, dimensions) for t in texts]

    @staticmethod
    def _openai_embed(texts: List[str], dimensions: int) -> List[List[float]]:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"[EmbeddingService] OpenAI embed error: {e}, falling back to demo")
            return [EmbeddingService._demo_embed(t, dimensions) for t in texts]
