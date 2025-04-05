#!/usr/bin/env python3
"""Test script for the RAG query service implementation."""

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
from query_service.service import process_query
from query_service.query_enhancement import enhance_query, combine_enhancement_methods
from query_service.strategies import get_strategy


def test_query_enhancement():
    """Test different query enhancement methods."""
    logger.info("Testing query enhancement methods...")
    
    # Test query
    query = "What are the main applications of transformer models in NLP?"
    
    # Test different enhancement methods
    enhancement_methods = [
        "synonym_expansion",
        "hypothetical_document",
        "query_decomposition",
        "llm_rewrite"
    ]
    
    for method in enhancement_methods:
        logger.info(f"Testing enhancement method: {method}")
        try:
            start_time = time.time()
            enhanced = enhance_query(query, method)
            duration = time.time() - start_time
            
            if isinstance(enhanced, list):
                logger.info(f"Enhanced queries ({len(enhanced)}):")
                for i, eq in enumerate(enhanced):
                    logger.info(f"  {i+1}. {eq}")
            else:
                logger.info(f"Enhanced query: {enhanced}")
            
            logger.info(f"Enhancement completed in {duration:.2f}s")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error testing {method}: {str(e)}")
    
    # Test combined enhancement
    logger.info("Testing combined enhancement methods...")
    try:
        start_time = time.time()
        combined = combine_enhancement_methods(query, ["llm_rewrite", "query_decomposition"])
        duration = time.time() - start_time
        
        logger.info(f"Combined enhancement results:")
        for method, result in combined.items():
            if isinstance(result, list):
                logger.info(f"  {method} ({len(result)}):")
                for i, r in enumerate(result):
                    logger.info(f"    {i+1}. {r}")
            else:
                logger.info(f"  {method}: {result}")
        
        logger.info(f"Combined enhancement completed in {duration:.2f}s")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing combined enhancement: {str(e)}")


def test_retrieval_strategies():
    """Test different retrieval strategies."""
    logger.info("Testing retrieval strategies...")
    
    # Test query
    query = "Explain the key differences between traditional databases and vector databases."
    
    # Test different strategies
    strategies = [
        "vector_search",
        "knowledge_graph",
        "hybrid"
    ]
    
    for strategy_name in strategies:
        logger.info(f"Testing strategy: {strategy_name}")
        try:
            start_time = time.time()
            
            # Create strategy instance
            strategy = get_strategy(strategy_name)
            
            # Retrieve documents
            documents, metadata = strategy.retrieve(query, top_k=3)
            
            duration = time.time() - start_time
            
            logger.info(f"Retrieved {len(documents)} documents:")
            for i, doc in enumerate(documents):
                logger.info(f"  {i+1}. {doc.id}: {doc.content[:100]}...")
            
            logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
            logger.info(f"Retrieval completed in {duration:.2f}s")
            logger.info("-" * 50)
        except Exception as e:
            logger.error(f"Error testing {strategy_name}: {str(e)}")


def test_full_query_processing():
    """Test the complete query processing pipeline."""
    logger.info("Testing full query processing...")
    
    # Test queries
    test_queries = [
        "What is the difference between dense and sparse embeddings?",
        "How do transformer models handle long context?",
        "Explain the concept of attention in neural networks."
    ]
    
    # Test configurations
    test_configs = [
        {
            "name": "Basic Vector Search",
            "enhancement_method": None,
            "strategy_name": "vector_search"
        },
        {
            "name": "Enhanced Vector Search",
            "enhancement_method": "llm_rewrite",
            "strategy_name": "vector_search"
        },
        {
            "name": "Knowledge Graph Retrieval",
            "enhancement_method": None,
            "strategy_name": "knowledge_graph"
        },
        {
            "name": "Hybrid Search with Query Decomposition",
            "enhancement_method": "query_decomposition",
            "strategy_name": "hybrid",
            "strategy_params": {"hybrid_weight": 0.6}
        }
    ]
    
    for query in test_queries:
        logger.info(f"Processing query: {query}")
        
        for config in test_configs:
            logger.info(f"Testing configuration: {config['name']}")
            try:
                start_time = time.time()
                
                result = process_query(
                    query=query,
                    enhancement_method=config.get("enhancement_method"),
                    strategy_name=config.get("strategy_name"),
                    strategy_params=config.get("strategy_params"),
                    top_k=3,
                    return_documents=True
                )
                
                duration = time.time() - start_time
                
                logger.info(f"Response: {result['response']}")
                logger.info(f"Retrieved {len(result.get('documents', []))} documents")
                logger.info(f"Processing completed in {duration:.2f}s")
                logger.info("-" * 50)
            except Exception as e:
                logger.error(f"Error processing query with {config['name']}: {str(e)}")


if __name__ == "__main__":
    """Run the test functions."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if test_name == "enhancement":
            test_query_enhancement()
        elif test_name == "strategies":
            test_retrieval_strategies()
        elif test_name == "full":
            test_full_query_processing()
        else:
            logger.error(f"Unknown test name: {test_name}")
            logger.info("Available tests: enhancement, strategies, full")
    else:
        # Run all tests
        logger.info("Running all tests...")
        test_query_enhancement()
        test_retrieval_strategies()
        test_full_query_processing()
        logger.info("All tests completed!") 