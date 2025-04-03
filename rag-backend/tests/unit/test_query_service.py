import unittest
from unittest.mock import patch, MagicMock

# assuming this is the structure, update imports if needed
from query_service.service import process_query
from query_service.strategies import VectorSearchStrategy, KnowledgeGraphStrategy, HybridStrategy
from query_service.query_enhancement import enhance_query


class TestQueryService(unittest.TestCase):
    @patch('query_service.service.get_strategy')
    @patch('query_service.service.enhance_query')
    @patch('query_service.service.save_query_log')
    @patch('query_service.service.generate_response')
    @patch('query_service.service.calculate_metrics')
    def test_process_query_happy_path(self, mock_calculate_metrics, mock_generate_response,
                                   mock_save_log, mock_enhance_query, mock_get_strategy):
        # arrange
        query = "What is RAG?"
        document_ids = ["doc1", "doc2"]
        strategy_name = "vector_search"
        parameters = {"chunk_count": 5}
        
        mock_enhance_query.return_value = "What is Retrieval Augmented Generation (RAG)?"
        mock_strategy = MagicMock()
        mock_strategy.retrieve.return_value = ["context1", "context2"]
        mock_get_strategy.return_value = mock_strategy
        
        mock_generate_response.return_value = "RAG is a technique that combines retrieval with generation."
        mock_calculate_metrics.return_value = {
            "relevance": 0.95,
            "completeness": 0.9,
            "latency_ms": 150
        }
        
        # act
        result = process_query(query, document_ids, strategy_name, parameters)
        
        # assert
        mock_enhance_query.assert_called_once_with(query, parameters.get("enhancement_method"))
        mock_get_strategy.assert_called_once_with(strategy_name)
        mock_strategy.retrieve.assert_called_once()
        mock_generate_response.assert_called_once()
        mock_calculate_metrics.assert_called_once()
        mock_save_log.assert_called_once()
        
        self.assertEqual(result["query"], query)
        self.assertEqual(result["result"], "RAG is a technique that combines retrieval with generation.")
        self.assertEqual(result["context"], ["context1", "context2"])
        self.assertEqual(result["metrics"]["relevance"], 0.95)
    
    @patch('query_service.strategies.vector_search.search_vectors')
    def test_vector_search_strategy_happy_path(self, mock_search_vectors):
        # arrange
        query = "What is RAG?"
        document_ids = ["doc1", "doc2"]
        parameters = {"chunk_count": 5, "model": "text-embedding-ada-002"}
        
        mock_chunks = [
            {"id": "chunk1", "text": "RAG is Retrieval Augmented Generation", "score": 0.92},
            {"id": "chunk2", "text": "RAG combines retrieval with generation", "score": 0.88},
            {"id": "chunk3", "text": "RAG enhances LLM responses with external knowledge", "score": 0.85},
        ]
        mock_search_vectors.return_value = mock_chunks
        
        # act
        result = VectorSearchStrategy(query, document_ids, parameters).retrieve()
        
        # assert
        mock_search_vectors.assert_called_once_with(query, document_ids, parameters.get("model"),
                                                 parameters.get("chunk_count"))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "RAG is Retrieval Augmented Generation")
    
    @patch('query_service.query_enhancement.rewrite_with_llm')
    def test_enhance_query_happy_path(self, mock_rewrite):
        # arrange
        query = "What is RAG?"
        enhancement_method = "llm_rewrite"
        mock_rewrite.return_value = "What is Retrieval Augmented Generation (RAG) and how does it work?"
        
        # act
        result = enhance_query(query, enhancement_method)
        
        # assert
        mock_rewrite.assert_called_once_with(query)
        self.assertEqual(result, "What is Retrieval Augmented Generation (RAG) and how does it work?") 