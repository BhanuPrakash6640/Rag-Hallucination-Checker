"""Number check: NLI models are weak at numbers, so we verify them with simple rules.

If a claim contains a number (amount, date, count, percentage) that never appears
in the evidence, the claim is very likely made up, e.g. "February 7, 2022" when the
source says "February 7, 1945".
"""

import re

_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000,
    "lakh": 100_000, "crore": 10_000_000, "dozen": 12,
}
# "one" is left out on purpose: "one of the benefits" is not a number claim.

_DIGITS = re.compile(r"(?<![\w.])\d+(?:,\d{2,3})*(?:\.\d+)?")
_WORDS = re.compile(r"\b(" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)


def _normalize(token: str) -> str:
    token = token.replace(",", "")
    whole, _, fraction = token.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


def extract_numbers(text: str) -> set[str]:
    numbers = {_normalize(m.group(0)) for m in _DIGITS.finditer(text)}
    numbers |= {str(_NUMBER_WORDS[m.group(1).lower()]) for m in _WORDS.finditer(text)}
    return numbers


def missing_numbers(claim: str, evidence: str) -> list[str]:
    """Numbers in the claim that do not appear anywhere in the evidence."""
    found = extract_numbers(evidence)
    return sorted(n for n in extract_numbers(claim) if n not in found)
