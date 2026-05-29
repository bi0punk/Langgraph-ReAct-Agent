from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""

    docs_path: Path = Path("data/docs")
    vectorstore_path: Path = Path("vectorstore/faiss_index")
    rag_top_k: int = 4
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_mode: str = "both"
    max_rewrites: int = 2

    checkpointer: str = "memory"
    checkpoint_db: str = "data/checkpoints.sqlite"

    log_level: str = "INFO"

    thread_id: str = "cli-session-1"
    embedding_model: str = "intfloat/e5-small"


settings = Settings()
