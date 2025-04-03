import unittest
from unittest.mock import patch, MagicMock

# assuming this is the structure, update imports if needed
from evaluation_service.service import calculate_metrics, run_benchmark
from evaluation_service.metrics import calculate_relevance, calculate_truthfulness, calculate_completeness


class TestEvaluationService(unittest.TestCase):
    @patch('evaluation_service.service.calculate_relevance')
    @patch('evaluation_service.service.calculate_truthfulness')
    @patch('evaluation_service.service.calculate_completeness')
    @patch('evaluation_service.service.get_latency')
    @patch('evaluation_service.service.count_tokens')
    def test_calculate_metrics_happy_path(self, mock_count_tokens, mock_get_latency,
                                       mock_calc_completeness, mock_calc_truthfulness,
                                       mock_calc_relevance):
        # arrange
        query = "What is RAG?"
        context = ["RAG is Retrieval Augmented Generation", 
                  "RAG combines retrieval with generation"]
        response = "RAG (Retrieval Augmented Generation) is a technique that combines retrieval of external knowledge with text generation."
        
        mock_calc_relevance.return_value = 0.95
        mock_calc_truthfulness.return_value = 0.98
        mock_calc_completeness.return_value = 0.92
        mock_get_latency.return_value = 150
        mock_count_tokens.return_value = {"context": 25, "response": 18}
        
        # act
        result = calculate_metrics(query, context, response)
        
        # assert
        mock_calc_relevance.assert_called_once_with(query, context, response)
        mock_calc_truthfulness.assert_called_once_with(context, response)
        mock_calc_completeness.assert_called_once_with(query, response)
        mock_get_latency.assert_called_once()
        mock_count_tokens.assert_called_once_with(context, response)
        
        self.assertEqual(result["relevance"], 0.95)
        self.assertEqual(result["truthfulness"], 0.98)
        self.assertEqual(result["completeness"], 0.92)
        self.assertEqual(result["latency_ms"], 150)
        self.assertEqual(result["tokens"]["context"], 25)
        self.assertEqual(result["tokens"]["response"], 18)
    
    @patch('evaluation_service.metrics.openai.chat.completions.create')
    def test_calculate_relevance_happy_path(self, mock_openai_chat):
        # arrange
        query = "What is RAG?"
        context = ["RAG is Retrieval Augmented Generation", 
                  "RAG combines retrieval with generation"]
        response = "RAG (Retrieval Augmented Generation) is a technique that combines retrieval of external knowledge with text generation."
        
        # mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.95"
        mock_openai_chat.return_value = mock_response
        
        # act
        result = calculate_relevance(query, context, response)
        
        # assert
        mock_openai_chat.assert_called_once()
        self.assertEqual(result, 0.95)
    
    @patch('evaluation_service.service.get_benchmark_questions')
    @patch('evaluation_service.service.process_query')
    @patch('evaluation_service.service.save_benchmark_results')
    def test_run_benchmark_happy_path(self, mock_save_results, mock_process_query, 
                                   mock_get_questions):
        # arrange
        document_ids = ["doc1", "doc2"]
        strategy_name = "vector_search"
        parameters = {"chunk_count": 5}
        
        benchmark_questions = [
            {"id": "q1", "question": "What is RAG?"},
            {"id": "q2", "question": "How does RAG work?"}
        ]
        mock_get_questions.return_value = benchmark_questions
        
        mock_results = [
            {
                "query_id": "q1",
                "result": "RAG is Retrieval Augmented Generation",
                "metrics": {"relevance": 0.95, "completeness": 0.9}
            },
            {
                "query_id": "q2",
                "result": "RAG works by retrieving content and then generating text",
                "metrics": {"relevance": 0.92, "completeness": 0.88}
            }
        ]
        mock_process_query.side_effect = mock_results
        
        # act
        result = run_benchmark(document_ids, strategy_name, parameters)
        
        # assert
        mock_get_questions.assert_called_once()
        self.assertEqual(mock_process_query.call_count, 2)
        mock_save_results.assert_called_once()
        
        self.assertEqual(result["total_questions"], 2)
        self.assertEqual(result["avg_relevance"], 0.935)  # (0.95 + 0.92) / 2
        self.assertEqual(result["avg_completeness"], 0.89)  # (0.9 + 0.88) / 2 