import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# assuming this is the structure, update imports if needed
from embedding_service.service import generate_embeddings
from embedding_service.models import EmbeddingModel


class TestEmbeddingService(unittest.TestCase):
    @patch('embedding_service.service.get_embedding_model')
    @patch('embedding_service.service.get_chunks')
    @patch('embedding_service.service.save_embeddings')
    @patch('embedding_service.service.update_document_status')
    def test_generate_embeddings_happy_path(self, mock_update_status, mock_save_embeddings, 
                                        mock_get_chunks, mock_get_model):
        # arrange
        document_id = "test-doc-id"
        model_name = "text-embedding-ada-002"
        chunks = [
            {"id": "chunk1", "text": "This is the first chunk", "position": 0},
            {"id": "chunk2", "text": "This is the second chunk", "position": 1},
        ]
        mock_get_chunks.return_value = chunks
        
        mock_model = MagicMock(spec=EmbeddingModel)
        mock_model.dimensions = 1536
        # generate sample vector
        sample_vector = np.random.rand(1536).tolist()
        mock_model.embed.side_effect = [sample_vector, sample_vector]
        mock_get_model.return_value = mock_model
        
        # act
        result = generate_embeddings(document_id, model_name)
        
        # assert
        mock_get_chunks.assert_called_once_with(document_id)
        mock_get_model.assert_called_once_with(model_name)
        self.assertEqual(mock_model.embed.call_count, 2)
        mock_save_embeddings.assert_called_once()
        mock_update_status.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["document_id"], document_id)
        self.assertEqual(result["model"], model_name)
        self.assertEqual(result["chunks_processed"], 2)
    
    @patch('embedding_service.models.openai.embedding.create')
    def test_embedding_model_embed_text_happy_path(self, mock_openai_embed):
        # arrange
        model = EmbeddingModel(model_id="text-embedding-ada-002", dimensions=1536)
        text = "This is a test text for embedding"
        
        # mock OpenAI response
        mock_embedding = np.random.rand(1536).tolist()
        mock_openai_response = {
            "data": [{"embedding": mock_embedding}]
        }
        mock_openai_embed.return_value = mock_openai_response
        
        # act
        result = model.embed(text)
        
        # assert
        mock_openai_embed.assert_called_once()
        self.assertEqual(len(result), 1536)  # check dimensions
        self.assertEqual(result, mock_embedding) 