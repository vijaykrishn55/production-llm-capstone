from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    max_upload_mb: int = 20
    max_chunk_chars: int = 1200
    chunk_overlap_chars: int = 180
    default_top_k: int = 5
    max_top_k: int = 12
    rrf_k: int = 60
    max_agent_steps: int = 5
    rate_limit_per_minute: int = 120
    sqlite_path: str = "data/capstone.db"
    corpus_path: str = "data/corpus.json"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "capstone_chunks"
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    vllm_timeout_seconds: float = 90.0
    max_new_tokens: int = 700
    hf_token: str | None = None


@lru_cache

def get_settings() -> Settings:
    return Settings()
