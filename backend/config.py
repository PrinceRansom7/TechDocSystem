"""
Application configuration using pydantic-settings.
All values are loaded from .env (or environment variables).
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM (Groq) ──────────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # ── Embeddings (sentence-transformers, local/free) ────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # downloaded on first run

    # ── ChromaDB ────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection: str = "techdocs"

    # ── Web Search ───────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    top_k: int = 5
    max_retries: int = 3

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_strategy: str = "hybrid"
    max_chunk_chars: int = 1500
    overlap_chars: int = 150
    min_chunk_chars: int = 200

    # ── Paths ─────────────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    log_level: str = "INFO"

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
