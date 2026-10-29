"""Answer generation with any OpenAI-compatible LLM (Ollama, Groq, OpenAI, vLLM, ...)."""

import time

from .config import Settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the numbered context passages. "
    "Cite the passages you use like [1] or [2]. If the context does not contain the answer, say "
    "that the documents do not mention it. Keep answers short: 2-5 sentences."
)


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, start=1))


class Generator:
    def __init__(self, settings: Settings | None = None):
        from openai import OpenAI

        self.settings = settings or Settings()
        self.client = OpenAI(base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key, timeout=120)

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 512) -> tuple[str, dict]:
        """Send messages, return (text, info) where info has latency and token usage."""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.llm_model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = response.usage
        info = {
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return (response.choices[0].message.content or "").strip(), info

    def answer(self, question: str, chunks: list[dict], temperature: float = 0.2) -> tuple[str, dict]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {question}"},
        ]
        return self.chat(messages, temperature=temperature)
