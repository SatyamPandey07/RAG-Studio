"""
Hybrid Search Service — BM25 sparse search + dense cosine similarity + RRF fusion.

The BM25 index is maintained in-memory per collection (reloaded from DB on demand).
RRF (Reciprocal Rank Fusion) combines dense and sparse rankings.
"""
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict


class BM25:
    """Simple BM25 implementation (no external dependency)."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(doc.split()) for doc in corpus) / max(1, self.n)
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_term_freqs: List[Dict[str, int]] = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _build_index(self):
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            freq: Dict[str, int] = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self.doc_term_freqs.append(dict(freq))
            for token in freq:
                self.doc_freqs[token] += 1

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        doc_len = sum(self.doc_term_freqs[doc_idx].values())
        score = 0.0
        for term in tokens:
            if term not in self.doc_term_freqs[doc_idx]:
                continue
            tf = self.doc_term_freqs[doc_idx][term]
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = [(i, self.score(query, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"doc_idx": idx, "score": round(s, 4)} for idx, s in scores[:top_k] if s > 0]


class HybridSearch:
    # In-memory BM25 index cache: collection_name → BM25 instance
    _bm25_index: Dict[str, BM25] = {}
    # Store corpus metadata alongside index: collection_name → list of {text, source, chunk_id, ...}
    _corpus_meta: Dict[str, List[Dict]] = {}

    @classmethod
    def build_bm25_index(cls, collection_name: str, chunks: List[Dict[str, Any]]):
        """Build/rebuild BM25 index for a collection from a list of chunk dicts."""
        texts = [c["text"] for c in chunks]
        cls._bm25_index[collection_name] = BM25(texts)
        cls._corpus_meta[collection_name] = chunks

    @classmethod
    def sparse_search(
        cls,
        collection_name: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """BM25 keyword search."""
        if collection_name not in cls._bm25_index:
            return []
        bm25 = cls._bm25_index[collection_name]
        meta = cls._corpus_meta[collection_name]
        hits = bm25.search(query, top_k)
        results = []
        for hit in hits:
            idx = hit["doc_idx"]
            if idx < len(meta):
                results.append({
                    **meta[idx],
                    "score": hit["score"],
                    "search_type": "sparse"
                })
        return results

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        RRF: score = Σ weight / (k + rank)
        Combines dense and sparse result lists into a single ranked list.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        rrf_meta: Dict[str, Dict] = {}

        def _key(r):
            return r.get("id") or r.get("chunk_id") or r.get("text", "")[:80]

        for rank, result in enumerate(dense_results):
            key = _key(result)
            rrf_scores[key] += dense_weight / (k + rank + 1)
            rrf_meta[key] = {**result, "search_type": "hybrid"}

        for rank, result in enumerate(sparse_results):
            key = _key(result)
            rrf_scores[key] += sparse_weight / (k + rank + 1)
            if key not in rrf_meta:
                rrf_meta[key] = {**result, "search_type": "hybrid"}

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        fused = []
        for i, key in enumerate(sorted_keys):
            item = rrf_meta[key]
            item["rrf_score"] = round(rrf_scores[key], 5)
            item["rank"] = i + 1
            fused.append(item)
        return fused

    @classmethod
    def has_index(cls, collection_name: str) -> bool:
        return collection_name in cls._bm25_index
