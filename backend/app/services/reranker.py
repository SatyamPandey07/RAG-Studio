"""
Reranker Service — cross-encoder reranking to improve retrieval precision.

In DEMO_MODE: boosts scores using query-term overlap heuristic.
With COHERE_API_KEY: uses Cohere Rerank v3.
"""
import random
from typing import List, Dict, Any, Optional
from app.core.config import settings


class Reranker:

    @staticmethod
    def rerank(
        query: str,
        results: List[Dict[str, Any]],
        model: str = "cohere",
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved chunks by relevance to the query.
        Returns list sorted by rerank_score descending.
        """
        if not results:
            return []

        top_n = top_n or len(results)

        if settings.DEMO_MODE or not settings.COHERE_API_KEY:
            return Reranker._demo_rerank(query, results, top_n)

        if model == "cohere" and settings.COHERE_API_KEY:
            return Reranker._cohere_rerank(query, results, top_n)

        return Reranker._demo_rerank(query, results, top_n)

    @staticmethod
    def _demo_rerank(
        query: str,
        results: List[Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Heuristic reranking: boost score by query term overlap.
        Simulates a cross-encoder quality improvement.
        """
        import re
        query_terms = set(re.findall(r'\w+', query.lower()))

        reranked = []
        for i, r in enumerate(results):
            text = r.get("text", "").lower()
            text_terms = set(re.findall(r'\w+', text))
            overlap = len(query_terms & text_terms)
            # Boost: 0.3 per overlapping term, capped at 0.5
            boost = min(0.5, overlap * 0.3)
            original_score = r.get("rrf_score") or r.get("score") or 0.5
            rerank_score = min(0.999, original_score + boost + random.uniform(-0.01, 0.02))
            reranked.append({
                **r,
                "original_score": round(original_score, 4),
                "rerank_score": round(rerank_score, 4),
                "original_rank": i + 1,
            })

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, r in enumerate(reranked):
            r["reranked_rank"] = i + 1

        return reranked[:top_n]

    @staticmethod
    def _cohere_rerank(
        query: str,
        results: List[Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Real Cohere Rerank API call."""
        try:
            import cohere
            co = cohere.Client(settings.COHERE_API_KEY)
            docs = [r.get("text", "") for r in results]
            response = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=docs,
                top_n=top_n,
            )
            reranked = []
            for item in response.results:
                original = results[item.index]
                reranked.append({
                    **original,
                    "original_rank": item.index + 1,
                    "original_score": original.get("score", 0),
                    "rerank_score": round(item.relevance_score, 4),
                    "reranked_rank": len(reranked) + 1,
                })
            return reranked
        except Exception as e:
            print(f"[Reranker] Cohere error: {e}, falling back to demo rerank")
            return Reranker._demo_rerank(query, results, top_n)


