"""
LLM Service — multi-provider chat completions via LiteLLM abstraction.

In DEMO_MODE: returns realistic simulated answers.
With API keys: calls real LLM (Google Gemini, OpenAI GPT-4o, Anthropic Claude, etc.)
"""
import time
import random
from typing import List, Dict, Any, Optional
from app.core.config import settings

LLM_PROVIDERS = {
    "gemini/gemini-2.0-flash": {
        "label": "Gemini 2.0 Flash",
        "provider": "google",
        "cost_per_1k_input": 0.00010,
        "cost_per_1k_output": 0.00040,
        "context_window": 1_000_000,
        "supports_streaming": True,
    },
    "gemini/gemini-2.5-pro": {
        "label": "Gemini 2.5 Pro",
        "provider": "google",
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.01000,
        "context_window": 2_000_000,
        "supports_streaming": True,
    },
    "gpt-4o": {
        "label": "GPT-4o",
        "provider": "openai",
        "cost_per_1k_input": 0.00250,
        "cost_per_1k_output": 0.01000,
        "context_window": 128_000,
        "supports_streaming": True,
    },
    "gpt-4o-mini": {
        "label": "GPT-4o Mini",
        "provider": "openai",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.00060,
        "context_window": 128_000,
        "supports_streaming": True,
    },
    "claude-3-5-sonnet-20241022": {
        "label": "Claude 3.5 Sonnet",
        "provider": "anthropic",
        "cost_per_1k_input": 0.00300,
        "cost_per_1k_output": 0.01500,
        "context_window": 200_000,
        "supports_streaming": True,
    },
    "groq/llama-3.1-70b-versatile": {
        "label": "Llama 3.1 70B (Groq)",
        "provider": "groq",
        "cost_per_1k_input": 0.00059,
        "cost_per_1k_output": 0.00079,
        "context_window": 131_072,
        "supports_streaming": True,
    },
}

DEMO_ANSWERS = [
    "Based on the retrieved context, {topic} is well-documented in the provided knowledge base. The system retrieved {chunks} relevant chunks with an average similarity score of {score:.2f}. According to the documents, {topic} works by leveraging key principles found in the indexed content.",
    "The retrieved context provides clear information about this topic. Drawing from {chunks} relevant passages, the answer is grounded in your uploaded documents and demonstrates strong faithfulness to the source material.",
    "According to your knowledge base ({chunks} chunks retrieved), this question relates to concepts that are thoroughly covered. The retrieval system identified high-confidence matches (avg score: {score:.2f}) from your indexed documents.",
]


class LLMService:

    @staticmethod
    def get_providers() -> dict:
        return LLM_PROVIDERS

    @staticmethod
    def chat(
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: str,
        model: str = "gemini/gemini-2.0-flash",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Run a RAG chat call.
        Returns: answer, token_usage, cost_usd, latency_ms
        """
        start = time.time()

        context_text = "\n\n---\n\n".join([
            f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        ])

        if settings.DEMO_MODE or not LLMService._has_key_for_model(model):
            # Simulate a realistic LLM response
            time.sleep(random.uniform(0.2, 0.6))
            avg_score = sum(c.get("score", 0.8) for c in context_chunks) / max(1, len(context_chunks))
            topic = " ".join(query.split()[:4]) + "..."
            template = random.choice(DEMO_ANSWERS)
            answer = template.format(
                topic=topic,
                chunks=len(context_chunks),
                score=avg_score
            )
            input_tokens = len(system_prompt.split()) + len(context_text.split()) + len(query.split())
            output_tokens = len(answer.split())
            latency = int((time.time() - start) * 1000) + random.randint(80, 200)
            cost = (input_tokens / 1000 * 0.0001) + (output_tokens / 1000 * 0.0004)
            return {
                "answer": answer,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": round(cost, 6),
                "latency_ms": latency,
                "demo_mode": True,
            }

        # Real LLM call via LiteLLM
        try:
            import litellm
            LLMService._configure_litellm(model)

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]

            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            usage = response.usage
            model_info = LLM_PROVIDERS.get(model, {})
            cost = (
                (usage.prompt_tokens / 1000 * model_info.get("cost_per_1k_input", 0.001)) +
                (usage.completion_tokens / 1000 * model_info.get("cost_per_1k_output", 0.004))
            )
            latency = int((time.time() - start) * 1000)

            return {
                "answer": answer,
                "model": model,
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6),
                "latency_ms": latency,
                "demo_mode": False,
            }

        except Exception as e:
            # Fallback to demo mode on error
            print(f"[LLMService] Error calling {model}: {e}, falling back to demo")
            latency = int((time.time() - start) * 1000)
            return {
                "answer": f"I retrieved {len(context_chunks)} relevant chunks for your query. (Note: LLM call failed: {str(e)[:100]})",
                "model": model,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": latency,
                "demo_mode": True,
                "error": str(e),
            }

    @staticmethod
    def _has_key_for_model(model: str) -> bool:
        if "gemini" in model:
            return bool(settings.GOOGLE_API_KEY)
        if "gpt" in model or "openai" in model:
            return bool(settings.OPENAI_API_KEY)
        if "claude" in model or "anthropic" in model:
            return bool(settings.ANTHROPIC_API_KEY)
        return False

    @staticmethod
    def _configure_litellm(model: str):
        try:
            import litellm
            if settings.GOOGLE_API_KEY:
                import os
                os.environ["GEMINI_API_KEY"] = settings.GOOGLE_API_KEY
            if settings.OPENAI_API_KEY:
                import os
                os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
            if settings.ANTHROPIC_API_KEY:
                import os
                os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY
        except Exception:
            pass
