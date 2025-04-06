import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# assuming this is the structure, update imports if needed
from query_service.service import process_query
from query_service.strategies import VectorSearchStrategy, KnowledgeGraphStrategy, HybridStrategy
from query_service.query_enhancement import enhance_query
from vector_store.models import Document


class TestQueryService(unittest.TestCase):
    @patch('query_service.service.get_strategy')
    @patch('query_service.service.enhance_query')
    @patch('query_service.service.save_query_log')
    @patch('query_service.service.generate_response')
    @patch('query_service.service.evaluate_response')
    def test_process_query_happy_path(self, mock_evaluate_response, mock_generate_response,
                                   mock_save_log, mock_enhance_query, mock_get_strategy):
        # arrange
        query = "What is RAG?"
        document_ids = ["doc1", "doc2"]
        strategy_name = "vector_search"
        parameters = {"chunk_count": 5}
        
        mock_enhance_query.return_value = "What is Retrieval Augmented Generation (RAG)?"
        mock_strategy = MagicMock()
        mock_strategy.retrieve.return_value = (
            [Document(id="1", content="context1"), Document(id="2", content="context2")],
            {"strategy": "vector_search"}
        )
        mock_get_strategy.return_value = mock_strategy
        
        mock_generate_response.return_value = "RAG is a technique that combines retrieval with generation."
        mock_evaluate_response.return_value = {
            "relevance": 0.95,
            "completeness": 0.9,
            "latency_ms": 150
        }
        
        # act
        result = process_query(query, document_ids, strategy_name, parameters)
        
        # assert
        mock_enhance_query.assert_called_once_with(query, None, None)
        mock_get_strategy.assert_called_once_with(strategy_name, **parameters, document_ids=document_ids)
        mock_strategy.retrieve.assert_called_once()
        mock_generate_response.assert_called_once()
        mock_evaluate_response.assert_called_once()
        mock_save_log.assert_called_once()
        
        self.assertEqual(result["query"], query)
        self.assertEqual(result["result"], "RAG is a technique that combines retrieval with generation.")
        self.assertEqual(len(result["context"]), 2)
        self.assertEqual(result["metrics"]["relevance"], 0.95)
    
    @patch('query_service.strategies.get_embedding')
    @patch('query_service.strategies.get_top_documents')
    def test_vector_search_strategy_happy_path(self, mock_get_top_documents, mock_get_embedding):
        # arrange
        query = "What is RAG?"
        enhanced_query = None
        top_k = 3
        
        # Create mock embedding vector
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_get_embedding.return_value = mock_embedding
        
        # Create mock documents
        mock_docs = [
            Document(id="chunk1", content="RAG is Retrieval Augmented Generation", score=0.92),
            Document(id="chunk2", content="RAG combines retrieval with generation", score=0.88),
            Document(id="chunk3", content="RAG enhances LLM responses with external knowledge", score=0.85),
        ]
        mock_get_top_documents.return_value = mock_docs
        
        # Create strategy instance
        strategy = VectorSearchStrategy()
        
        # act
        documents, metadata = strategy.retrieve(query, enhanced_query, top_k)
        
        # assert
        mock_get_embedding.assert_called_once_with(query)
        mock_get_top_documents.assert_called_once()
        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].content, "RAG is Retrieval Augmented Generation")
        self.assertEqual(metadata["strategy"], "vector_search")
    
    @patch('query_service.query_enhancement.QueryEnhancer.llm_rewrite')
    def test_enhance_query_happy_path(self, mock_llm_rewrite):
        # arrange
        query = "What is RAG?"
        enhancement_method = "llm_rewrite"
        mock_llm_rewrite.return_value = "What is Retrieval Augmented Generation (RAG) and how does it work?"
        
        # act
        result = enhance_query(query, enhancement_method)
        
        # assert
        mock_llm_rewrite.assert_called_once_with(query)
        self.assertEqual(result, "What is Retrieval Augmented Generation (RAG) and how does it work?") 