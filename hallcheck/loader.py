"""Load documents from a folder and cut them into overlapping chunks."""

from pathlib import Path

from .text_utils import split_sentences, word_count

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(docs_dir: str) -> list[dict]:
    """Return [{"source": file name, "text": full text}] for every supported file."""
    folder = Path(docs_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Documents folder not found: {folder.resolve()}")

    documents = []
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        text = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            documents.append({"source": path.name, "text": text})
    if not documents:
        raise ValueError(f"No .txt, .md or .pdf files with text found in {folder.resolve()}")
    return documents


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Pack whole sentences into chunks of about `chunk_size` words.

    Consecutive chunks share roughly `overlap` words so facts that sit on a
    boundary are not lost.
    """
    sentences = [s for s in split_sentences(text) if not s.startswith("#") or word_count(s) > 6]
    sentences = [s.lstrip("#").strip() for s in sentences]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        n = word_count(sentence)
        if current and current_words + n > chunk_size:
            chunks.append(" ".join(current))
            # keep the last few sentences as overlap
            kept: list[str] = []
            kept_words = 0
            for prev in reversed(current):
                if kept_words + word_count(prev) > overlap:
                    break
                kept.insert(0, prev)
                kept_words += word_count(prev)
            current, current_words = kept, kept_words
        current.append(sentence)
        current_words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 120, overlap: int = 30) -> list[dict]:
    chunks = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({"id": len(chunks), "source": doc["source"], "chunk": i, "text": text})
    return chunks
