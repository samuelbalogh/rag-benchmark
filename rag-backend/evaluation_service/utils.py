"""Utility functions for evaluation service."""

import time
from typing import Dict, List, Any

from common.logging import get_logger

logger = get_logger(__name__)


def get_latency() -> int:
    """Simulate latency measurement for testing.
    
    Returns:
        Latency in milliseconds
    """
    # For testing purposes, return a consistent value
    # In a real system, this would calculate actual latency
    return 150


def count_tokens(context: List[str], response: str) -> Dict[str, int]:
    """Count tokens in context and response.
    
    Args:
        context: List of context passages
        response: Generated response
        
    Returns:
        Dictionary with token counts
    """
    # Simple token counting by splitting on whitespace
    # In a real system, this would use the tokenizer from the LLM
    context_text = " ".join(context)
    context_tokens = len(context_text.split())
    response_tokens = len(response.split())
    
    return {
        "context": context_tokens,
        "response": response_tokens
    }


def get_benchmark_questions() -> List[Dict[str, Any]]:
    """Get benchmark questions for evaluation.
    
    Returns:
        List of benchmark questions
    """
    # In a real system, these would come from a database or file
    return [
        {"id": "q1", "question": "What is RAG?"},
        {"id": "q2", "question": "How does RAG work?"}
    ]


def process_query(query_id: str, query: str, document_ids: List[str], 
                strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Process a query with the specified retrieval strategy.
    
    Args:
        query_id: Query ID
        query: Query text
        document_ids: List of document IDs to search
        strategy_name: Name of the retrieval strategy
        parameters: Strategy parameters
        
    Returns:
        Query processing result
    """
    # In a real system, this would use the retrieval and generation pipeline
    # For testing, return mock results
    return {
        "query_id": query_id,
        "result": f"Mock result for {query}",
        "metrics": {
            "relevance": 0.95,
            "completeness": 0.9
        }
    }


def save_benchmark_results(results: Dict[str, Any]) -> None:
    """Save benchmark results.
    
    Args:
        results: Benchmark results
    """
    # In a real system, this would save to database or file
    logger.info(f"Saving benchmark results: {len(results['detailed_results'])} queries processed") 