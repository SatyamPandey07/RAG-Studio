from typing import List, Optional
import hashlib
import random
from app.core.config import settings

class Embedder:
    PROVIDERS = {
        "google": {
            "models": ["text-embedding-004"],
            "dimensions": 768,
            "cost_per_1k": 0.00004
        },
        "openai": {
            "models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            "dimensions": 1536,
            "cost_per_1k": 0.0001
        },
        "voyage": {
            "models": ["voyage-2", "voyage-large-2"],
            "dimensions": 1024,
            "cost_per_1k": 0.0001
        },
        "jina": {
            "models": ["jina-embeddings-v2-base-en"],
            "dimensions": 768,
            "cost_per_1k": 0.00009
        }
    }

    @classmethod
    def get_mock_embedding(cls, text: str, dimensions: int = 768) -> List[float]:
        """
        Generates a consistent pseudo-random vector based on text hash
        to support offline/demo mode without requiring API keys.
        """
        hash_val = hashlib.sha256(text.encode('utf-8')).hexdigest()
        # Seed generator with hash value
        seed = int(hash_val[:8], 16)
        rng = random.Random(seed)
        
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
        
        # L2 Normalize the vector
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
            
        return vector

    @classmethod
    def get_embedding(cls, text: str, provider: str, model: str) -> List[float]:
        provider = provider.lower()
        model_meta = cls.PROVIDERS.get(provider, {"dimensions": 768})
        dimensions = model_meta.get("dimensions", 768)
        
        # Check if API Key is available to call live models
        if provider == "google" and settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                result = genai.embed_content(
                    model=f"models/{model}",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception:
                pass # Fallback to mock
                
        elif provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from litellm import embedding
                response = embedding(
                    model=f"openai/{model}",
                    input=[text]
                )
                return response['data'][0]['embedding']
            except Exception:
                pass # Fallback to mock
                
        # Return pseudo-random but consistent embedding vector
        return cls.get_mock_embedding(text, dimensions=dimensions)
        
    @classmethod
    def get_provider_details(cls) -> dict:
        return cls.PROVIDERS
