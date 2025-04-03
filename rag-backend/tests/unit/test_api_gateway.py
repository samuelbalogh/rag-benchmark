import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

try:
    from api_gateway.main import app
except ImportError:
    # Mark the module as skippable if imports fail
    pytest.skip("Required modules not available", allow_module_level=True)


class TestApiGateway(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.api_key = "test-api-key"
        self.headers = {"X-API-Key": self.api_key}
    
    @patch('api_gateway.routers.documents.process_document')
    @patch('api_gateway.routers.documents.save_document_metadata')
    def test_upload_document_happy_path(self, mock_save_metadata, mock_process_document):
        # arrange
        mock_save_metadata.return_value = {"id": "test-doc-id"}
        mock_process_document.delay = MagicMock()
        
        # create test file
        test_file_content = b"This is a test document"
        
        # act
        response = self.client.post(
            "/api/v1/documents",
            headers=self.headers,
            files={"file": ("test.pdf", test_file_content, "application/pdf")},
            data={"name": "Test Document", "description": "Test description"}
        )
        
        # assert
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["id"], "test-doc-id")
        mock_save_metadata.assert_called_once()
        mock_process_document.delay.assert_called_once()
    
    @patch('api_gateway.routers.query.process_query')
    def test_process_query_happy_path(self, mock_process_query):
        # arrange
        mock_response = {
            "query_id": "test-query-id",
            "result": "This is a test result",
            "context": ["context1", "context2"],
            "metrics": {
                "relevance": 0.95,
                "completeness": 0.85,
                "latency_ms": 150
            }
        }
        mock_process_query.return_value = mock_response
        
        # act
        response = self.client.post(
            "/api/v1/query",
            headers=self.headers,
            json={
                "query": "What is RAG?",
                "strategy": "vector_search",
                "document_ids": ["doc1", "doc2"],
                "parameters": {"chunk_count": 5}
            }
        )
        
        # assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_response)
        mock_process_query.assert_called_once()
    
    @patch('api_gateway.routers.config.get_embedding_models')
    def test_get_embedding_models_happy_path(self, mock_get_embedding_models):
        # arrange
        mock_models = [
            {"id": "text-embedding-ada-002", "name": "Ada", "dimensions": 1536},
            {"id": "text-embedding-3-small", "name": "Small", "dimensions": 1536},
            {"id": "text-embedding-3-large", "name": "Large", "dimensions": 3072}
        ]
        mock_get_embedding_models.return_value = mock_models
        
        # act
        response = self.client.get("/api/v1/config/embedding-models", headers=self.headers)
        
        # assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), mock_models)
        mock_get_embedding_models.assert_called_once() 