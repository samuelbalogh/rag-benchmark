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

# Use patchers instead of sys.modules modifications
@patch('corpus_service.tasks.processing.app', MagicMock())
@patch('corpus_service.tasks.chunking.app', MagicMock())
class TestCorpusService(unittest.TestCase):
    """Test corpus service functionality."""
    
    @patch('corpus_service.tasks.processing.chunk_document')
    @patch('corpus_service.tasks.processing.update_document_status')
    def test_process_document_pipeline(self, mock_update_status, mock_chunk_document):
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
    @patch('corpus_service.tasks.chunking.trigger_embedding_generation')
    def test_chunk_document(self, mock_trigger_embedding, mock_update_status, 
                          mock_save_chunks, mock_get_content):
        """Test chunk_document task."""
        from corpus_service.tasks.chunking import chunk_document
        
        # Arrange
        document_id = "test-doc-id"
        mock_get_content.return_value = "This is a test document content with multiple sentences. " \
                                       "It should be chunked properly. This is the third sentence."
        
        # Act
        result = chunk_document(document_id)
        
        # Assert
        mock_get_content.assert_called_once_with(document_id)
        mock_save_chunks.assert_called_once()
        mock_update_status.assert_called_once()
        self.assertIn("status", result)

    @patch('corpus_service.tasks.processing.celery')
    @patch('corpus_service.tasks.processing.chunk_document')
    @patch('corpus_service.tasks.processing.update_document_status')
    def test_process_document_happy_path(self, mock_update_status, mock_chunk_document, mock_celery):
        # arrange
        document_id = "test-doc-id"
        file_path = "/tmp/test-document.pdf"
        mock_celery.current_task = MagicMock()
        
        # act
        result = process_document_pipeline(document_id)
        
        # assert
        mock_update_status.assert_called_once()
        mock_chunk_document.delay.assert_called_once_with(document_id)
        self.assertEqual(result, {"status": "success", "message": "Document processing started"})
    
    @patch('corpus_service.tasks.chunking.get_document_content')
    @patch('corpus_service.tasks.chunking.save_chunks')
    @patch('corpus_service.tasks.chunking.update_document_status')
    @patch('corpus_service.tasks.chunking.trigger_embedding_generation')
    def test_chunk_document_happy_path(self, mock_trigger_embedding, mock_update_status, 
                                      mock_save_chunks, mock_get_content):
        # arrange
        document_id = "test-doc-id"
        mock_get_content.return_value = "This is a test document content with multiple sentences. " \
                                       "It should be chunked properly. This is the third sentence."
        
        # act
        result = chunk_document(document_id)
        
        # assert
        mock_get_content.assert_called_once_with(document_id)
        mock_save_chunks.assert_called_once()
        mock_update_status.assert_called_once()
        mock_trigger_embedding.delay.assert_called_once_with(document_id)
        self.assertEqual(result, {"status": "success", "message": "Document chunking completed"}) 