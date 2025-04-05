#!/usr/bin/env python3
"""Test script for the RAG orchestration service implementation."""

import os
import sys
import json
import time
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import required modules
from orchestration_service.service import (
    process_query_with_strategy,
    compare_strategies,
    process_complex_query
)
from query_service.service import process_query


def test_query_with_strategies():
    """Test processing queries with different predefined strategies."""
    logger.info("Testing query processing with different strategies...")
    
    # Test queries
    test_queries = [
        "What is RAG in the context of language models?",
        "Compare the performance of vector databases and traditional databases for semantic search."
    ]
    
    # Test with different strategy keys
    strategy_keys = ["vector", "knowledge_graph", "hybrid", "enhanced_vector"]
    
    for query in test_queries:
        logger.info(f"Processing query: {query}")
        
        for strategy_key in strategy_keys:
            logger.info(f"Testing with strategy: {strategy_key}")
            try:
                start_time = time.time()
                
                result = process_query_with_strategy(
                    query=query,
                    strategy_key=strategy_key,
                    return_documents=True
                )
                
                duration = time.time() - start_time
                
                logger.info(f"Response: {result['response'][:100]}...")
                logger.info(f"Retrieved {len(result.get('documents', []))} documents")
                logger.info(f"Processing completed in {duration:.2f}s")
                logger.info("-" * 50)
            except Exception as e:
                logger.error(f"Error processing query with {strategy_key}: {str(e)}")


def test_strategy_comparison():
    """Test comparing multiple RAG strategies."""
    logger.info("Testing strategy comparison...")
    
    # Test queries
    test_queries = [
        "Explain how knowledge graphs can enhance retrieval-augmented generation."
    ]
    
    for query in test_queries:
        logger.info(f"Comparing strategies for query: {query}")
        try:
            start_time = time.time()
            
            comparison = compare_strategies(
                query=query,
                strategy_keys=["vector", "knowledge_graph", "hybrid"],
                return_documents=False
            )
            
            duration = time.time() - start_time
            
            logger.info(f"Compared {len(comparison['strategies'])} strategies")
            
            # Print results for each strategy
            for strategy, result in comparison['results'].items():
                if 'error' in result:
                    logger.error(f"Strategy {strategy} error: {result['error']}")
                else:
                    logger.info(f"Strategy {strategy} response: {result['response'][:100]}...")
            
            # Print metrics comparison
            if 'metrics_comparison' in comparison:
                logger.info("Metrics comparison:")
                for metric, details in comparison['metrics_comparison'].items():
                    logger.info(f"  {metric}: {details}")
            
            logger.info(f"Comparison completed in {duration:.2f}s")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error comparing strategies: {str(e)}")


def test_complex_query_processing():
    """Test processing complex queries with adaptive strategy selection."""
    logger.info("Testing complex query processing...")
    
    # Test queries with different characteristics
    test_queries = [
        "What is the difference between dense and sparse embeddings?",  # Simple descriptive
        "Who created transformer models and when were they first introduced?",  # Entity-focused
        "Explain the relationship between attention mechanisms and transformers in deep learning.",  # Relationship
        "What are vector databases? How do they work? When should they be used instead of traditional databases?",  # Multiple questions
    ]
    
    for query in test_queries:
        logger.info(f"Processing complex query: {query}")
        try:
            start_time = time.time()
            
            result = process_complex_query(
                query=query,
                return_documents=True
            )
            
            duration = time.time() - start_time
            
            # Print query analysis
            if 'query_analysis' in result:
                logger.info(f"Query analysis: {json.dumps(result['query_analysis'], indent=2)}")
                logger.info(f"Selected strategy: {result['query_analysis']['recommended_strategy']}")
            
            logger.info(f"Response: {result['response'][:100]}...")
            logger.info(f"Retrieved {len(result.get('documents', []))} documents")
            logger.info(f"Processing completed in {duration:.2f}s")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error processing complex query: {str(e)}")


def test_custom_strategy_parameters():
    """Test processing query with custom strategy parameters."""
    logger.info("Testing custom strategy parameters...")
    
    query = "What are the advantages of using LangChain for RAG applications?"
    
    # Test custom parameters
    custom_params = {
        "enhancement_method": "combined",
        "enhancement_methods": ["llm_rewrite", "query_decomposition"],
        "top_k": 10,
        "llm_model": "gpt-4",
    }
    
    logger.info(f"Processing query with custom parameters: {query}")
    try:
        start_time = time.time()
        
        result = process_query_with_strategy(
            query=query,
            strategy_key="vector",
            override_params=custom_params,
            return_documents=True
        )
        
        duration = time.time() - start_time
        
        logger.info(f"Response: {result['response'][:100]}...")
        logger.info(f"Retrieved {len(result.get('documents', []))} documents")
        logger.info(f"Used enhancement methods: {custom_params['enhancement_methods']}")
        logger.info(f"Processing completed in {duration:.2f}s")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error processing query with custom parameters: {str(e)}")


if __name__ == "__main__":
    """Run the test functions."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if test_name == "strategies":
            test_query_with_strategies()
        elif test_name == "comparison":
            test_strategy_comparison()
        elif test_name == "complex":
            test_complex_query_processing()
        elif test_name == "custom":
            test_custom_strategy_parameters()
        else:
            logger.error(f"Unknown test name: {test_name}")
            logger.info("Available tests: strategies, comparison, complex, custom")
    else:
        # Run all tests
        logger.info("Running all tests...")
        test_query_with_strategies()
        test_strategy_comparison()
        test_complex_query_processing()
        test_custom_strategy_parameters()
        logger.info("All tests completed!") 