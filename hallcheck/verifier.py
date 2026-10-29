"""Step 3: the full checker. Answer + source passages in, a verdict per claim out.

Pipeline for each claim:
  1. Rank evidence windows (2-sentence slices of the passages) by keyword overlap
     and keep the top few. This keeps the NLI model fast: it only reads what matters.
  2. Run the NLI model on (window, claim) pairs.
  3. Decide: SUPPORTED / CONTRADICTED / UNSUPPORTED, then apply the number check.
"""

import time
from dataclasses import asdict, dataclass, field

from .claims import Claim, extract_claims
from .numbers import extract_numbers, missing_numbers
from .text_utils import keywords, split_sentences

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNSUPPORTED = "unsupported"
SKIPPED = "skipped"


@dataclass
class ClaimResult:
    claim: str
    verdict: str
    entailment: float = 0.0
    contradiction: float = 0.0
    evidence: str = ""
    evidence_source: str = ""
    missing_numbers: list[str] = field(default_factory=list)
    reason: str = ""
    sentence_index: int = 0

    @property
    def risk(self) -> float:
        """0 = clearly supported, 1 = clearly hallucinated. Used for threshold tuning."""
        if self.verdict == SKIPPED:
            return 0.0
        risk = 1.0 - self.entailment
        if self.missing_numbers:
            risk = max(risk, 0.9)
        return round(risk, 4)


@dataclass
class CheckResult:
    claims: list[ClaimResult]
    latency_ms: float = 0.0
    nli_pairs: int = 0

    @property
    def checked(self) -> list[ClaimResult]:
        return [c for c in self.claims if c.verdict != SKIPPED]

    @property
    def hallucinated(self) -> bool:
        return any(c.verdict in (CONTRADICTED, UNSUPPORTED) for c in self.checked)

    @property
    def risk_score(self) -> float:
        return max((c.risk for c in self.checked), default=0.0)

    def counts(self) -> dict[str, int]:
        out = {SUPPORTED: 0, CONTRADICTED: 0, UNSUPPORTED: 0, SKIPPED: 0}
        for c in self.claims:
            out[c.verdict] += 1
        return out

    def to_dict(self) -> dict:
        return {
            "hallucinated": self.hallucinated,
            "risk_score": self.risk_score,
            "latency_ms": round(self.latency_ms, 1),
            "nli_pairs": self.nli_pairs,
            "counts": self.counts(),
            "claims": [{**asdict(c), "risk": c.risk} for c in self.claims],
        }


def _as_passage(p) -> tuple[str, str]:
    if isinstance(p, dict):
        return p.get("text", ""), p.get("source", "")
    return str(p), ""


class HallucinationChecker:
    def __init__(
        self,
        scorer,
        entail_threshold: float = 0.5,
        contra_threshold: float = 0.5,
        windows_per_claim: int = 6,
        window_size: int = 2,
        check_numbers: bool = True,
        split_compound: bool = True,
    ):
        """`scorer` is anything with .predict([(premise, hypothesis)]) -> [{"entailment",..}], e.g. NLIScorer."""
        self.scorer = scorer
        self.entail_threshold = entail_threshold
        self.contra_threshold = contra_threshold
        self.windows_per_claim = windows_per_claim
        self.window_size = window_size
        self.check_numbers = check_numbers
        self.split_compound = split_compound

    @classmethod
    def from_settings(cls, settings=None) -> "HallucinationChecker":
        from .config import Settings
        from .nli import NLIScorer

        s = settings or Settings()
        return cls(
            NLIScorer(s.nli_model, device=s.device),
            entail_threshold=s.entail_threshold,
            contra_threshold=s.contra_threshold,
            windows_per_claim=s.windows_per_claim,
            window_size=s.window_size,
            check_numbers=s.check_numbers,
        )

    # ---------- evidence ----------
    def build_windows(self, passages) -> list[dict]:
        """Slide a window of `window_size` sentences over every passage."""
        windows = []
        for p in passages:
            text, source = _as_passage(p)
            sentences = split_sentences(text)
            if not sentences:
                continue
            size = max(1, self.window_size)
            for start in range(0, max(1, len(sentences) - size + 1)):
                chunk = " ".join(sentences[start : start + size])
                windows.append({"text": chunk, "source": source, "kw": keywords(chunk), "nums": extract_numbers(chunk)})
        return windows

    def _rank(self, claim: str, windows: list[dict]) -> list[int]:
        claim_kw = keywords(claim)
        claim_nums = extract_numbers(claim)

        def score(w):
            overlap = len(claim_kw & w["kw"]) / max(1, len(claim_kw))
            return overlap + 0.5 * len(claim_nums & w["nums"])

        ranked = sorted(range(len(windows)), key=lambda i: score(windows[i]), reverse=True)
        return ranked[: self.windows_per_claim]

    # ---------- main entry ----------
    def check(self, answer: str, passages, claims: list[Claim] | None = None) -> CheckResult:
        start = time.perf_counter()
        claims = claims if claims is not None else extract_claims(answer, self.split_compound)
        windows = self.build_windows(passages)
        all_evidence = " ".join(_as_passage(p)[0] for p in passages)

        # Collect every (window, claim) pair first, then score them in one batched call.
        pairs, owners = [], []
        candidates: dict[int, list[int]] = {}
        for ci, claim in enumerate(claims):
            if claim.skipped or not windows:
                continue
            candidates[ci] = self._rank(claim.text, windows)
            for wi in candidates[ci]:
                pairs.append((windows[wi]["text"], claim.text))
                owners.append((ci, wi))
        scores = self.scorer.predict(pairs) if pairs else []

        per_claim: dict[int, list[tuple[int, dict]]] = {}
        for (ci, wi), s in zip(owners, scores):
            per_claim.setdefault(ci, []).append((wi, s))

        results = []
        for ci, claim in enumerate(claims):
            if claim.skipped:
                results.append(ClaimResult(claim.text, SKIPPED, reason=claim.skip_reason, sentence_index=claim.sentence_index))
                continue
            results.append(self._decide(claim, per_claim.get(ci, []), windows, all_evidence))

        return CheckResult(results, (time.perf_counter() - start) * 1000, len(pairs))

    def _decide(self, claim: Claim, scored: list[tuple[int, dict]], windows: list[dict], all_evidence: str) -> ClaimResult:
        if not scored:
            return ClaimResult(claim.text, UNSUPPORTED, reason="no evidence text", sentence_index=claim.sentence_index)

        best_e_wi, best_e = max(scored, key=lambda x: x[1]["entailment"])
        best_c_wi, best_c = max(scored, key=lambda x: x[1]["contradiction"])
        e, c = best_e["entailment"], best_c["contradiction"]

        if e >= self.entail_threshold:
            verdict, wi, reason = SUPPORTED, best_e_wi, "evidence supports the claim"
        elif c >= self.contra_threshold:
            verdict, wi, reason = CONTRADICTED, best_c_wi, "evidence says something different"
        else:
            verdict, wi, reason = UNSUPPORTED, scored[0][0], "no passage supports the claim"

        missing = missing_numbers(claim.text, all_evidence) if self.check_numbers else []
        if missing:
            reason = f"number(s) {', '.join(missing)} not found in the sources"
            if verdict == SUPPORTED:
                verdict = UNSUPPORTED

        return ClaimResult(
            claim=claim.text,
            verdict=verdict,
            entailment=round(e, 4),
            contradiction=round(c, 4),
            evidence=windows[wi]["text"],
            evidence_source=windows[wi]["source"],
            missing_numbers=missing,
            reason=reason,
            sentence_index=claim.sentence_index,
        )
