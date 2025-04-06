"""Basic tests to verify the testing infrastructure is working."""

import os
import sys
import pytest


def test_python_version():
    """Check that the Python version is at least 3.10."""
    version_info = sys.version_info
    assert version_info.major >= 3
    assert version_info.major == 3 and version_info.minor >= 10


def test_package_structure():
    """Test that the package structure is correct."""
    # Test that required directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert os.path.exists(os.path.join(base_dir, "common")), "common directory not found"
    assert os.path.exists(os.path.join(base_dir, "vector_store")), "vector_store directory not found"
    assert os.path.exists(os.path.join(base_dir, "orchestration_service")), "orchestration_service directory not found"
    assert os.path.exists(os.path.join(base_dir, "evaluation_service")), "evaluation_service directory not found"


def test_imports():
    """Test that the basic imports work."""
    from common.logging import get_logger
    from vector_store.models import Document
    from vector_store.service import VectorStore, get_top_documents
    from vector_store.sparse import keyword_search, BM25Retriever
    
    # Test creating a document
    doc = Document(id="test1", content="This is a test document", metadata={"key": "value"})
    assert doc.id == "test1"
    assert doc.content == "This is a test document"
    assert doc.metadata == {"key": "value"} 