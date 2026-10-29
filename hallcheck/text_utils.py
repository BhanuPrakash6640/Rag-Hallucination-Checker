"""Small text helpers: sentence splitting and keyword extraction. No ML needed."""

import re

_ABBREVIATIONS = [
    "e.g.", "i.e.", "etc.", "vs.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
    "Inc.", "Ltd.", "Co.", "Corp.", "No.", "St.", "Jan.", "Feb.", "Mar.", "Apr.",
    "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.", "U.S.", "U.K.",
    "approx.", "Rs.", "a.m.", "p.m.", "Gov.", "Sen.", "Rep.", "Gen.", "Lt.", "Col.", "Capt.",
    "Sgt.", "Ave.", "Mt.", "Ft.", "Pres.", "Rev.", "Hon.",
]
_DOT = "․"  # placeholder for protected dots
# Match abbreviations only as whole words ("Rs." yes, the end of "leaders." no), keeping original case.
_ABBR_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True)) + ")",
    re.IGNORECASE,
)

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = set(
    """a an the and or but if then else of to in on at by for with from as is are was were be been
    being it its this that these those there their they them he she his her we our you your i me my
    do does did doing have has had having not no nor so than too very can will just should would could
    may might must shall about above after again against all any because before below between both
    during each few further here how into more most other out over own same some such through under
    until up what when where which while who whom why also only per via""".split()
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Each line/bullet is treated as its own unit."""
    if not text:
        return []
    sentences: list[str] = []
    for line in re.split(r"\n+", text):
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        protected = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _DOT), line)
        for part in _SENT_END.split(protected):
            part = part.replace(_DOT, ".").strip()
            if part:
                sentences.append(part)
    return sentences


def keywords(text: str) -> set[str]:
    """Lower-cased content words (stopwords removed) used for cheap evidence ranking."""
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def word_count(text: str) -> int:
    return len(text.split())
