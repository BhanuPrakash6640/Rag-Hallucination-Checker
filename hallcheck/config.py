"""All settings in one place. Values come from environment variables or a .env file."""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return type(default)(value)


@dataclass
class Settings:
    # Retrieval
    docs_dir: str = field(default_factory=lambda: _env("DOCS_DIR", "data/docs"))
    index_dir: str = field(default_factory=lambda: _env("INDEX_DIR", "data/index"))
    embed_model: str = field(
        default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: _env("CHUNK_SIZE", 120))  # words
    chunk_overlap: int = field(default_factory=lambda: _env("CHUNK_OVERLAP", 30))  # words
    top_k: int = field(default_factory=lambda: _env("TOP_K", 4))

    # Answer-writing LLM (any OpenAI-compatible server)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3.2:3b"))

    # Hallucination checker
    nli_model: str = field(
        default_factory=lambda: _env("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    )
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    entail_threshold: float = field(default_factory=lambda: _env("ENTAIL_THRESHOLD", 0.5))
    contra_threshold: float = field(default_factory=lambda: _env("CONTRA_THRESHOLD", 0.5))
    windows_per_claim: int = field(default_factory=lambda: _env("WINDOWS_PER_CLAIM", 6))
    window_size: int = field(default_factory=lambda: _env("WINDOW_SIZE", 2))
    check_numbers: bool = field(default_factory=lambda: _env("CHECK_NUMBERS", True))
