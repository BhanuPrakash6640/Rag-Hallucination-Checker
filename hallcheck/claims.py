"""Step 1 of the checker: break an answer into small, checkable claims."""

import json
import re
from dataclasses import dataclass

from .text_utils import split_sentences, word_count

_CITATION = re.compile(r"\s*\[\d+(?:\s*[,\-]\s*\d+)*\]")
_LEAD_IN = re.compile(
    r"^(?:according to|based on|as (?:stated|mentioned) in)\s+(?:the\s+)?(?:provided\s+|given\s+)?"
    r"(?:context|documents?|passages?|sources?|information|text|policy)[^,]{0,40},\s*",
    re.IGNORECASE,
)
_COMPOUND = re.compile(r",\s+(?:and|but|while|whereas)\s+|;\s+", re.IGNORECASE)

# Sentences that say "I don't know" are not hallucinations, so we skip them.
_ABSTAIN = re.compile(
    r"\b(?:not (?:mentioned|specified|provided|stated|available|covered)|"
    r"do(?:es)?(?: not|n't) (?:contain|provide|mention|specify|say|include|cover)|"
    r"no (?:information|details|mention)|(?:can ?not|can't|unable to) (?:find|answer|determine)|"
    r"i (?:don't|do not) know)\b",
    re.IGNORECASE,
)
# Polite filler that carries no facts.
_FILLER = re.compile(
    r"^(?:i hope (?:this|that) helps|let me know|feel free|please (?:let|contact|reach)|"
    r"is there anything else|happy to help|great question|sure[,!.]|certainly[,!.]|"
    r"here(?:'s| is| are) (?:a |an |the |my )?(?:\w+ )?(?:summary|answer|overview|response|list|breakdown|description))",
    re.IGNORECASE,
)


@dataclass
class Claim:
    text: str
    sentence_index: int
    skipped: bool = False
    skip_reason: str = ""


def clean_sentence(sentence: str) -> str:
    sentence = _CITATION.sub("", sentence)
    sentence = _LEAD_IN.sub("", sentence).strip()
    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def _split_compound(sentence: str, min_words: int = 4) -> list[str]:
    """Split "A, and B" / "A; B" into two claims when both halves are long enough.

    A ", and" that ends a list ("Delhi, Mumbai, and Pune") is not split: if the text
    before it already contains a comma, it is treated as a list, not two clauses.
    """
    pieces, last = [], 0
    for m in _COMPOUND.finditer(sentence):
        left = sentence[last : m.start()]
        if m.group(0).lstrip().startswith(";") or "," not in left:
            pieces.append(left)
            last = m.end()
    pieces.append(sentence[last:])
    pieces = [p.strip() for p in pieces if p.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and word_count(piece) < min_words:
            merged[-1] = f"{merged[-1]} and {piece}"
        else:
            merged.append(piece)
    if len(merged) > 1 and word_count(merged[0]) < min_words:
        merged[1] = f"{merged[0]} and {merged[1]}"
        merged = merged[1:]
    out = []
    for m in merged:
        m = m.rstrip(" .")
        out.append(m[0].upper() + m[1:] + "." if m else m)
    return out


def extract_claims(answer: str, split_compound: bool = True) -> list[Claim]:
    claims: list[Claim] = []
    for i, raw in enumerate(split_sentences(answer)):
        sentence = clean_sentence(raw)
        if not sentence:
            continue
        if sentence.endswith("?"):
            claims.append(Claim(sentence, i, True, "question"))
        elif _FILLER.match(sentence) or word_count(sentence) < 3 or sentence.rstrip(".").endswith(":"):
            claims.append(Claim(sentence, i, True, "no factual content"))
        elif _ABSTAIN.search(sentence):
            claims.append(Claim(sentence, i, True, "answer says the info is missing"))
        else:
            parts = _split_compound(sentence) if split_compound else [sentence]
            claims.extend(Claim(p, i) for p in parts)
    return claims


LLM_CLAIM_PROMPT = (
    "Break the answer below into short, self-contained factual claims. Each claim must make sense "
    "on its own (replace pronouns with names). Skip greetings and statements that the information is "
    'missing. Return ONLY a JSON list of strings.\n\nAnswer:\n"""{answer}"""'
)


def llm_extract_claims(answer: str, generator) -> list[Claim]:
    """Optional, more accurate but slower: ask an LLM to decompose the answer."""
    text, _ = generator.chat([{"role": "user", "content": LLM_CLAIM_PROMPT.format(answer=answer)}], temperature=0)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        items = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        items = []
    claims = [Claim(str(c).strip(), i) for i, c in enumerate(items) if str(c).strip()]
    return claims or extract_claims(answer)  # fall back to rule-based splitting
