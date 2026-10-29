"""hallcheck: a fast, claim-level hallucination checker for RAG systems."""

from .verifier import (
    CONTRADICTED,
    SKIPPED,
    SUPPORTED,
    UNSUPPORTED,
    CheckResult,
    ClaimResult,
    HallucinationChecker,
)

__all__ = [
    "HallucinationChecker",
    "CheckResult",
    "ClaimResult",
    "SUPPORTED",
    "CONTRADICTED",
    "UNSUPPORTED",
    "SKIPPED",
]
