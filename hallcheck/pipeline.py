"""Glue: retrieve -> generate -> check."""

import time

from .config import Settings
from .verifier import HallucinationChecker


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, checker: HallucinationChecker | None = None):
        from .generator import Generator
        from .retriever import Retriever

        self.settings = settings or Settings()
        self.retriever = Retriever(self.settings).load()
        self.generator = Generator(self.settings)
        self.checker = checker or HallucinationChecker.from_settings(self.settings)

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        chunks = self.retriever.search(question)
        t1 = time.perf_counter()
        answer, info = self.generator.answer(question, chunks)
        check = self.checker.check(answer, chunks)
        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
            "check": check,
            "timing_ms": {
                "retrieval": (t1 - t0) * 1000,
                "generation": info["latency_ms"],
                "checking": check.latency_ms,
            },
        }
