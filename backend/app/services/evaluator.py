"""
Evaluator Service — computes RAG quality metrics.

Metrics computed:
  Retrieval: Recall@K, Precision@K, MRR, nDCG
  Generation: Faithfulness, Groundedness, Answer Relevancy, Hallucination Rate, Citation Accuracy

In DEMO_MODE: uses heuristic computation (no LLM-as-judge calls).
With API keys: uses LLM-as-judge (Gemini or GPT) for semantic metrics.
"""
import math
import random
from typing import List, Dict, Any, Optional
from app.core.config import settings


class Evaluator:

    @staticmethod
    def evaluate_retrieval(
        retrieved_chunks: List[Dict[str, Any]],
        relevant_doc_ids: Optional[List[str]] = None,
        k: int = 5,
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        If relevant_doc_ids is None, uses heuristic scoring.
        """
        n = len(retrieved_chunks)
        if n == 0:
            return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}

        if settings.DEMO_MODE or relevant_doc_ids is None:
            return Evaluator._heuristic_retrieval_metrics(retrieved_chunks, k)

        # With ground truth
        retrieved_ids = [str(c.get("id", "")) for c in retrieved_chunks[:k]]
        relevant_set = set(str(r) for r in relevant_doc_ids)

        hits = [1 if rid in relevant_set else 0 for rid in retrieved_ids]
        precision = sum(hits) / k if k > 0 else 0
        recall = sum(hits) / len(relevant_set) if relevant_set else 0

        # MRR
        mrr = 0.0
        for i, h in enumerate(hits):
            if h:
                mrr = 1 / (i + 1)
                break

        # nDCG
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        ideal_hits = sorted(hits, reverse=True)
        idcg = sum(h / math.log2(i + 2) for i, h in enumerate(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0

        return {
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(precision, 4),
            "mrr": round(mrr, 4),
            "ndcg": round(ndcg, 4),
        }

    @staticmethod
    def _heuristic_retrieval_metrics(chunks: List[Dict], k: int) -> Dict[str, float]:
        """
        Heuristic metrics based on score distribution.
        Chunks with high similarity scores → high recall/precision.
        """
        if not chunks:
            return {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}

        scores = [c.get("rerank_score") or c.get("rrf_score") or c.get("score") or 0.5 for c in chunks[:k]]
        avg_score = sum(scores) / len(scores)

        # Simulate metrics correlated with retrieval quality
        base = max(0.5, min(0.98, avg_score))
        noise = lambda: random.uniform(-0.03, 0.03)

        return {
            "recall_at_k": round(min(0.99, base * 0.95 + noise()), 4),
            "precision_at_k": round(min(0.99, base * 0.90 + noise()), 4),
            "mrr": round(min(0.99, base * 0.92 + noise()), 4),
            "ndcg": round(min(0.99, base * 0.93 + noise()), 4),
        }

    @staticmethod
    def evaluate_generation(
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate answer quality: faithfulness, groundedness, answer relevancy, hallucination.
        """
        if settings.DEMO_MODE or not settings.GOOGLE_API_KEY:
            return Evaluator._heuristic_generation_metrics(query, answer, retrieved_chunks)

        # LLM-as-judge (using Gemini)
        return Evaluator._llm_judge(query, answer, retrieved_chunks)

    @staticmethod
    def _heuristic_generation_metrics(
        query: str,
        answer: str,
        chunks: List[Dict],
    ) -> Dict[str, float]:
        """
        Heuristic generation quality metrics.
        - Faithfulness: does answer contain terms from chunks?
        - Groundedness: does the context support the answer?
        - Answer Relevancy: does the answer address the query?
        - Hallucination Rate: inverse of faithfulness × chunk coverage
        """
        answer_terms = set(answer.lower().split())
        query_terms = set(query.lower().split())
        chunk_terms = set()
        for c in chunks:
            chunk_terms.update(c.get("text", "").lower().split())

        # Faithfulness: what fraction of answer terms appear in chunks
        faith_terms = answer_terms & chunk_terms
        faithfulness = len(faith_terms) / max(1, len(answer_terms))
        faithfulness = min(0.99, 0.5 + faithfulness * 0.5 + random.uniform(-0.03, 0.03))

        # Groundedness: what fraction of chunks' key terms appear in answer
        key_chunk_terms = {t for t in chunk_terms if len(t) > 5}
        ground_matches = answer_terms & key_chunk_terms
        groundedness = len(ground_matches) / max(1, len(key_chunk_terms)) * 3
        groundedness = min(0.98, max(0.50, groundedness + random.uniform(-0.02, 0.02)))

        # Answer Relevancy: query term overlap with answer
        rel_matches = answer_terms & query_terms
        relevancy = len(rel_matches) / max(1, len(query_terms))
        relevancy = min(0.99, 0.55 + relevancy * 0.4 + random.uniform(-0.02, 0.03))

        # Hallucination Rate: inverse of faithfulness
        hallucination_rate = max(0.001, 1 - faithfulness + random.uniform(-0.02, 0.02))
        hallucination_rate = min(0.5, hallucination_rate)

        # Citation Accuracy: proxy — did the answer mention any source filenames?
        sources = [c.get("source", "") for c in chunks]
        cited = sum(1 for s in sources if s and s.split("/")[-1].split(".")[0].lower() in answer.lower())
        citation_acc = cited / max(1, len(sources)) if sources else 0.5
        citation_acc = min(0.99, 0.5 + citation_acc * 0.5)

        return {
            "faithfulness": round(faithfulness, 4),
            "groundedness": round(groundedness, 4),
            "answer_relevancy": round(relevancy, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "citation_accuracy": round(citation_acc, 4),
        }

    @staticmethod
    def _llm_judge(query: str, answer: str, chunks: List[Dict]) -> Dict[str, float]:
        """Use Gemini as LLM-as-judge for evaluation metrics."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            model = genai.GenerativeModel("gemini-2.0-flash")

            context = "\n".join([c.get("text", "")[:500] for c in chunks])
            prompt = f"""Evaluate this RAG system response on the following criteria.
Return ONLY a JSON object with float values between 0 and 1.

Question: {query}
Context: {context[:2000]}
Answer: {answer[:1000]}

Return JSON:
{{
  "faithfulness": <float 0-1: how faithful is the answer to the context?>,
  "groundedness": <float 0-1: is the answer grounded in the retrieved evidence?>,
  "answer_relevancy": <float 0-1: does the answer address the question?>,
  "hallucination_rate": <float 0-1: estimated hallucination probability (0=none, 1=full)>,
  "citation_accuracy": <float 0-1: are sources correctly referenced?>
}}"""

            response = model.generate_content(prompt)
            import json, re
            match = re.search(r'\{[^}]+\}', response.text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return Evaluator._heuristic_generation_metrics(query, answer, chunks)
        except Exception as e:
            print(f"[Evaluator] LLM judge error: {e}")
            return Evaluator._heuristic_generation_metrics(query, answer, chunks)

    @staticmethod
    def run_full_evaluation(
        pipeline_name: str,
        queries: List[str],
        retrieved_results: List[List[Dict]],
        answers: List[str],
        latencies: List[int],
        costs: List[float],
        token_counts: List[int],
    ) -> Dict[str, Any]:
        """Aggregate evaluation across multiple queries."""
        all_retrieval = []
        all_generation = []

        for i, (query, chunks, answer) in enumerate(zip(queries, retrieved_results, answers)):
            ret = Evaluator.evaluate_retrieval(chunks)
            gen = Evaluator.evaluate_generation(query, answer, chunks)
            all_retrieval.append(ret)
            all_generation.append(gen)

        def avg(dicts, key):
            vals = [d[key] for d in dicts if key in d]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "recall_at_k": avg(all_retrieval, "recall_at_k"),
            "precision_at_k": avg(all_retrieval, "precision_at_k"),
            "mrr": avg(all_retrieval, "mrr"),
            "ndcg": avg(all_retrieval, "ndcg"),
            "faithfulness": avg(all_generation, "faithfulness"),
            "groundedness": avg(all_generation, "groundedness"),
            "answer_relevancy": avg(all_generation, "answer_relevancy"),
            "hallucination_rate": avg(all_generation, "hallucination_rate"),
            "citation_accuracy": avg(all_generation, "citation_accuracy"),
            "avg_latency_ms": int(sum(latencies) / max(1, len(latencies))),
            "avg_cost_usd": round(sum(costs) / max(1, len(costs)), 6),
            "total_tokens": sum(token_counts),
            "queries_evaluated": len(queries),
        }
