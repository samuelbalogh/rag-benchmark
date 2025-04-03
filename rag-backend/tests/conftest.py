"""Pytest configuration for the RAG Benchmark Platform."""

import os
import sys
import pytest
from typing import Dict, Any

# add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Fixture to set up test environment variables."""
    env_vars = {
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "test_db",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "OPENAI_API_KEY": "test_openai_key",
        "SECRET_KEY": "test_secret_key",
        "API_KEY_HEADER": "X-API-Key"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Fixture to provide a sample document."""
    return {
        "id": "test-doc-id",
        "name": "Test Document",
        "description": "A test document for unit tests",
        "created_at": "2023-01-01T00:00:00Z",
        "status": "processed",
        "metadata": {
            "file_type": "pdf",
            "page_count": 10,
            "word_count": 5000
        }
    }


@pytest.fixture
def sample_query() -> Dict[str, Any]:
    """Fixture to provide a sample query."""
    return {
        "query": "What is RAG?",
        "document_ids": ["test-doc-id"],
        "strategy": "vector_search",
        "parameters": {
            "chunk_count": 5,
            "model": "text-embedding-ada-002"
        }
    } 