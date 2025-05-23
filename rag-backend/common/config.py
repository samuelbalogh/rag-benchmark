"""Configuration module for the RAG benchmark platform."""

import os
from typing import Optional, Dict, Any
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # General settings
    app_name: str = "RAG Benchmark"
    debug: bool = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    
    # API settings
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", "8003"))
    
    # Database settings
    db_host: str = os.environ.get("DB_HOST", "localhost")
    db_port: int = int(os.environ.get("DB_PORT", "5432"))
    db_name: str = os.environ.get("DB_NAME", "rag_benchmark")
    db_user: str = os.environ.get("DB_USER", "postgres")
    db_password: str = os.environ.get("DB_PASSWORD", "postgres")
    
    # OpenAI settings
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    
    # Embedding settings
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_batch_size: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
    
    # Vector store settings
    vector_db_index_type: str = os.environ.get("VECTOR_DB_INDEX_TYPE", "hnsw")
    
    # LLM settings
    default_llm_model: str = os.environ.get("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    
    # Knowledge graph settings
    kg_storage_path: str = os.environ.get("KG_STORAGE_PATH", "./data/knowledge_graph")
    
    # Logging settings
    log_level: str = os.environ.get("LOG_LEVEL", "INFO")
    log_format: str = os.environ.get("LOG_FORMAT", "json")
    
    # Redis settings
    redis_host: str = os.environ.get("REDIS_HOST", "localhost")
    redis_port: str = os.environ.get("REDIS_PORT", "6379")
    
    # Celery settings
    celery_broker_url: str = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_result_backend: str = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    
    # Voyage API settings
    voyage_api_key: str = os.environ.get("VOYAGE_API_KEY", "your_voyage_api_key")
    
    # Document storage
    document_storage_path: str = os.environ.get("DOCUMENT_STORAGE_PATH", "/path/to/document/storage")
    
    # Security
    secret_key: str = os.environ.get("SECRET_KEY", "your_secret_key_here")
    api_key_header: str = os.environ.get("API_KEY_HEADER", "X-API-Key")
    
    # Use Pydantic V2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_db_url() -> str:
    """Get database URL from settings."""
    settings = get_settings()
    return f"postgresql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}" 