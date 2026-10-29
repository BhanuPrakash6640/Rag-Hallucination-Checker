"""Semantic search over document chunks using sentence embeddings + FAISS."""

import json
from pathlib import Path

import numpy as np

from .config import Settings
from .loader import chunk_documents, load_documents


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._model = None
        self.index = None
        self.chunks: list[dict] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embed_model)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return vectors.astype("float32")

    # ---------- building ----------
    def build(self, docs_dir: str | None = None) -> int:
        """Read documents, chunk them, embed the chunks and save the index. Returns chunk count."""
        import faiss

        s = self.settings
        documents = load_documents(docs_dir or s.docs_dir)
        self.chunks = chunk_documents(documents, s.chunk_size, s.chunk_overlap)
        vectors = self._embed([c["text"] for c in self.chunks])

        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine on normalized vectors
        self.index.add(vectors)
        self.save()
        return len(self.chunks)

    def save(self) -> None:
        import faiss

        folder = Path(self.settings.index_dir)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / "index.faiss"))
        meta = {"embed_model": self.settings.embed_model, "chunks": self.chunks}
        (folder / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- loading & searching ----------
    def load(self) -> "Retriever":
        import faiss

        folder = Path(self.settings.index_dir)
        index_path = folder / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found in {folder.resolve()}. Run `python build_index.py` first."
            )
        self.index = faiss.read_index(str(index_path))
        meta = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        if meta.get("embed_model") != self.settings.embed_model:
            raise ValueError(
                f"Index was built with {meta.get('embed_model')} but EMBED_MODEL is "
                f"{self.settings.embed_model}. Rebuild the index."
            )
        self.chunks = meta["chunks"]
        return self

    def search(self, query: str, k: int | None = None) -> list[dict]:
        if self.index is None:
            self.load()
        k = min(k or self.settings.top_k, len(self.chunks))
        scores, ids = self.index.search(self._embed([query]), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results
