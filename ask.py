"""Ask questions from the terminal and see which claims are supported.

Usage:
    python ask.py "How many paid leave days do I get?"
    python ask.py            # interactive mode
"""

import sys

from hallcheck.pipeline import RAGPipeline
from hallcheck.verifier import CONTRADICTED, SKIPPED, SUPPORTED, UNSUPPORTED

ICONS = {SUPPORTED: "OK  ", CONTRADICTED: "XX  ", UNSUPPORTED: "??  ", SKIPPED: "--  "}


def show(result: dict) -> None:
    check = result["check"]
    print(f"\nAnswer:\n{result['answer']}\n")
    print("Claim check  (OK = supported, XX = contradicted, ?? = not in sources, -- = skipped)")
    for c in check.claims:
        print(f"  {ICONS[c.verdict]}{c.claim}")
        if c.verdict in (CONTRADICTED, UNSUPPORTED):
            print(f"        reason: {c.reason}")
            if c.evidence:
                print(f"        closest evidence ({c.evidence_source}): {c.evidence[:160]}")
    t = result["timing_ms"]
    verdict = "POSSIBLE HALLUCINATION" if check.hallucinated else "All claims supported"
    print(f"\n{verdict} | risk {check.risk_score:.2f} | retrieval {t['retrieval']:.0f} ms, "
          f"generation {t['generation']:.0f} ms, checking {t['checking']:.0f} ms\n")


def main():
    print("Loading models (the first run downloads them, which can take a few minutes)...")
    pipeline = RAGPipeline()
    if len(sys.argv) > 1:
        show(pipeline.ask(" ".join(sys.argv[1:])))
        return
    while True:
        try:
            question = input("Question (empty to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        show(pipeline.ask(question))


if __name__ == "__main__":
    main()
