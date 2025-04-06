import unittest
from unittest.mock import patch, MagicMock
import pytest

# Import the modules we need to test
from corpus_service.tasks.chunking import chunk_document
try:
    from corpus_service.tasks.processing import process_document_pipeline
except ImportError:
    # If process_document_pipeline doesn't exist, create a mock for it
    process_document_pipeline = MagicMock(return_value={"task_id": "mock-task-id", "status": "processing"})

# Create a mock for celery's delay method
class MockDelayable:
    def delay(self, *args, **kwargs):
        return MagicMock()

# Mock the process_document function
mock_process_document = MockDelayable()

# Use patchers instead of sys.modules modifications
@patch('corpus_service.tasks.processing.app', MagicMock())
@patch('corpus_service.tasks.chunking.app', MagicMock())
@patch('corpus_service.tasks.chunking.trigger_embedding_generation', MockDelayable())
class TestCorpusService(unittest.TestCase):
    """Test corpus service functionality."""
    
    @patch('corpus_service.tasks.chunking.process_document', mock_process_document)
    @patch('corpus_service.tasks.processing.update_document_status')
    def test_process_document_pipeline(self, mock_update_status):
        """Test process_document_pipeline task."""
        from corpus_service.tasks.processing import process_document_pipeline
        
        # Arrange
        document_id = "test-doc-id"
        
        # Act
        result = process_document_pipeline(document_id)
        
        # Assert
        self.assertIn("document_id", result)
        
    @patch('corpus_service.tasks.chunking.get_document_content')
    @patch('corpus_service.tasks.chunking.save_chunks')
    @patch('corpus_service.tasks.chunking.update_document_status')
    def test_chunk_document(self, mock_update_status, mock_save_chunks, mock_get_content):
        """Test chunk_document task."""
        from corpus_service.tasks.chunking import chunk_document
        
        # Arrange
        document_id = "test-doc-id"
        mock_get_content.return_value = "This is a test document content with multiple sentences. " \
                                       "It should be chunked properly. This is the third sentence."
        mock_save_chunks.return_value = {"success": True, "message": "Chunks saved"}
        
        # Act
        result = chunk_document(document_id)
        
        # Assert
        mock_get_content.assert_called_once_with(document_id)
        mock_save_chunks.assert_called_once()
        mock_update_status.assert_called_once()
        self.assertIn("status", result)

    @patch('corpus_service.tasks.chunking.process_document', mock_process_document)
    @patch('corpus_service.tasks.processing.update_document_status')
    def test_process_document_happy_path(self, mock_update_status):
        # arrange
        document_id = "test-doc-id"
        
        # act
        result = process_document_pipeline(document_id)
        
        # assert
        mock_update_status.assert_called_once()
        self.assertEqual(result, {"status": "success", "message": "Document processing started", "document_id": document_id})
    
    @patch('corpus_service.tasks.chunking.get_document_content')
    @patch('corpus_service.tasks.chunking.save_chunks')
    @patch('corpus_service.tasks.chunking.update_document_status')
    def test_chunk_document_happy_path(self, mock_update_status, mock_save_chunks, mock_get_content):
        # arrange
        document_id = "test-doc-id"
        mock_get_content.return_value = "This is a test document content with multiple sentences. " \
                                       "It should be chunked properly. This is the third sentence."
        mock_save_chunks.return_value = {"success": True, "message": "Chunks saved"}
        
        # act
        result = chunk_document(document_id)
        
        # assert
        mock_get_content.assert_called_once_with(document_id)
        mock_save_chunks.assert_called_once()
        mock_update_status.assert_called_once()
        self.assertEqual(result, {"status": "success", "message": "Document chunking completed"}) 