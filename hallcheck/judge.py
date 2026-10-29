"""Baseline to compare against: ask a (big) LLM to judge the answer.

This is the slow, expensive approach most teams use today. Your checker should
reach similar accuracy at a fraction of the latency and cost.
"""

import json
import re

JUDGE_PROMPT = """You are a strict fact-checker. Decide whether the ANSWER contains any information that is not supported by the SOURCE, or that contradicts it.

SOURCE:
\"\"\"{source}\"\"\"

{question_block}ANSWER:
\"\"\"{answer}\"\"\"

Reply with ONLY this JSON:
{{"hallucinated": true or false, "unsupported_claims": ["..."]}}"""


class LLMJudge:
    def __init__(self, generator, max_source_chars: int = 12000):
        self.generator = generator
        self.max_source_chars = max_source_chars

    def judge(self, source: str, answer: str, question: str | None = None) -> dict:
        prompt = JUDGE_PROMPT.format(
            source=source[: self.max_source_chars],
            answer=answer,
            question_block=f"QUESTION:\n{question}\n\n" if question else "",
        )
        text, info = self.generator.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=400)
        parsed = _parse(text)
        return {**info, **parsed, "raw": text}


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "hallucinated": bool(data.get("hallucinated")),
                "unsupported_claims": data.get("unsupported_claims") or [],
                "parse_error": False,
            }
        except json.JSONDecodeError:
            pass
    # Fallback: look for the key words.
    lowered = text.lower()
    return {"hallucinated": '"hallucinated": true' in lowered or "hallucinated: true" in lowered,
            "unsupported_claims": [], "parse_error": True}
