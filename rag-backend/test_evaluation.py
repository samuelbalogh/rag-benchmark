#!/usr/bin/env python3
"""Test script for the RAG evaluation service implementation."""

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
from evaluation_service.service import (
    evaluate_response,
    evaluate_relevance,
    evaluate_faithfulness,
    evaluate_context_recall,
    evaluate_answer_relevancy,
    calculate_metrics
)
from vector_store.models import Document
from query_service.service import process_query


def create_test_document(content, doc_id="test-doc", metadata=None):
    """Create a test document with the given content."""
    if metadata is None:
        metadata = {"title": "Test Document"}
    
    return Document(
        id=doc_id,
        content=content,
        metadata=metadata,
        score=0.9
    )


def test_individual_metrics():
    """Test individual evaluation metrics."""
    logger.info("Testing individual evaluation metrics...")
    
    # Test data
    query = "What are the main components of a RAG system?"
    response = """
    A RAG (Retrieval-Augmented Generation) system has several main components:
    1. Document processing pipeline for ingesting and chunking documents
    2. Embedding model for converting text to vector representations
    3. Vector store for efficient storage and retrieval of document embeddings
    4. Retrieval component that finds relevant documents for a query
    5. Generation component (usually an LLM) that creates a response based on retrieved context
    Some advanced RAG systems also include knowledge graphs, query enhancement, and reranking.
    """
    
    # Create test documents
    documents = [
        create_test_document(
            """RAG systems have multiple components including document processing, embedding models, 
            vector stores, retrieval mechanisms, and generation models. The document processing 
            pipeline handles chunking and metadata extraction. Embedding models convert text to 
            vector representations. Vector stores provide efficient storage and similarity search.""",
            "doc1",
            {"title": "RAG Components Part 1"}
        ),
        create_test_document(
            """The retrieval component in RAG finds relevant documents for a query using vector 
            similarity or other search methods. The generation component is typically an LLM that 
            creates a coherent response based on the retrieved context. Advanced RAG systems 
            may include knowledge graphs for structured information retrieval.""",
            "doc2",
            {"title": "RAG Components Part 2"}
        )
    ]
    
    # Test relevance
    logger.info("Testing relevance evaluation...")
    try:
        result = evaluate_relevance(query, response)
        logger.info(f"Relevance score: {result.score:.2f}")
        logger.info(f"Explanation: {result.explanation}")
        logger.info(f"Metadata: {result.metadata}")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing relevance: {str(e)}")
    
    # Test faithfulness
    logger.info("Testing faithfulness evaluation...")
    try:
        result = evaluate_faithfulness(response, documents)
        logger.info(f"Faithfulness score: {result.score:.2f}")
        logger.info(f"Explanation: {result.explanation}")
        logger.info(f"Metadata: {result.metadata}")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing faithfulness: {str(e)}")
    
    # Test context recall
    ground_truth = """
    The main components of a RAG system include a document processing pipeline that handles 
    document ingestion, chunking, and metadata extraction. It also includes an embedding model 
    that converts text into vector representations. A vector store efficiently stores and 
    retrieves these embeddings. The retrieval component finds relevant documents for a query, 
    while the generation component (usually an LLM) creates a response based on the retrieved context.
    """
    
    logger.info("Testing context recall evaluation...")
    try:
        result = evaluate_context_recall(documents, ground_truth)
        logger.info(f"Context recall score: {result.score:.2f}")
        logger.info(f"Explanation: {result.explanation}")
        logger.info(f"Metadata: {result.metadata}")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing context recall: {str(e)}")
    
    # Test answer relevancy
    logger.info("Testing answer relevancy evaluation...")
    try:
        result = evaluate_answer_relevancy(query, response)
        logger.info(f"Answer relevancy score: {result.score:.2f}")
        logger.info(f"Explanation: {result.explanation}")
        logger.info(f"Metadata: {result.metadata}")
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing answer relevancy: {str(e)}")


def test_multiple_methods():
    """Test evaluating a response using multiple methods."""
    logger.info("Testing evaluation with multiple methods...")
    
    # Test data
    query = "How do vector databases compare to traditional databases for semantic search?"
    response = """
    Vector databases are optimized for semantic search by storing vector embeddings that capture 
    semantic meaning, while traditional databases primarily work with structured data and exact matches.
    
    Vector databases use specialized indexes like HNSW or IVF-Flat for approximate nearest neighbor 
    search, enabling efficient similarity matching. They can find semantically similar content even 
    when exact keywords don't match. In contrast, traditional databases rely on keyword-based indexes 
    and struggle with understanding semantic intent without additional processing.
    
    Vector databases typically scale better for high-dimensional vector operations but may have more 
    overhead for simple CRUD operations. Traditional databases excel at structured queries but require 
    extensions or additional services to support vector operations.
    
    For semantic search applications, vector databases generally provide better relevance and 
    performance, while traditional databases might be more appropriate when combined operations on 
    structured and unstructured data are needed.
    """
    
    # Create test documents
    documents = [
        create_test_document(
            """Vector databases are designed to store and efficiently query vector embeddings, which 
            are numerical representations of data that capture semantic meaning. Unlike traditional 
            databases that focus on exact matching and structured data, vector databases excel at 
            similarity search, finding items that are conceptually related even when keywords don't 
            match exactly. They use specialized indexing algorithms like HNSW or IVF-Flat to enable 
            fast approximate nearest neighbor search in high-dimensional spaces.""",
            "doc1",
            {"title": "Vector Databases Overview"}
        ),
        create_test_document(
            """Traditional databases like SQL databases store structured data in tables and use 
            B-tree indexes optimized for exact matching and range queries. They excel at CRUD 
            operations and joining related data but have limitations when it comes to semantic 
            similarity. For semantic search, traditional databases typically need to be extended 
            with specialized extensions or connected to external search services.""",
            "doc2",
            {"title": "Traditional Databases"}
        )
    ]
    
    # Ground truth answer
    ground_truth = """
    Vector databases are specifically designed for semantic search applications, storing data as 
    vector embeddings that represent semantic meaning. They use specialized indexes like HNSW 
    (Hierarchical Navigable Small World) or IVF-Flat to perform approximate nearest neighbor search, 
    which enables efficient similarity matching even in high-dimensional spaces.
    
    Traditional databases (SQL/NoSQL) primarily store structured data and use indexes optimized for 
    exact matching and range queries. They typically rely on B-tree or hash-based indexes that don't 
    efficiently support semantic similarity search.
    
    Key differences include:
    1. Query capability: Vector databases can find semantically similar content even when exact 
       keywords don't match, while traditional databases focus on exact matches.
    2. Performance: Vector databases are optimized for high-dimensional similarity search, providing 
       better performance for semantic queries.
    3. Scalability: Vector databases typically have specialized architectures for scaling vector 
       operations, while traditional databases may struggle with high-dimensional data.
    4. Integration: Traditional databases often have more mature ecosystems and tools, while vector 
       databases are more specialized.
    
    For applications heavily reliant on semantic search, vector databases provide significant 
    advantages in relevance and performance, though some traditional databases now offer vector 
    extensions (like PostgreSQL with pgvector) to bridge this gap.
    """
    
    # Test multiple evaluation methods
    logger.info("Evaluating response with multiple methods...")
    try:
        methods = ["relevance", "faithfulness", "context_recall", "answer_relevancy"]
        results = evaluate_response(query, response, documents, ground_truth, methods)
        
        for method, result in results.items():
            logger.info(f"{method.capitalize()} score: {result.score:.2f}")
            logger.info(f"Explanation: {result.explanation}")
            logger.info("-" * 30)
        
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing multiple evaluation methods: {str(e)}")


def test_with_real_rag_output():
    """Test evaluation metrics with real RAG output."""
    logger.info("Testing evaluation with real RAG output...")
    
    # Process a query with RAG
    query = "What are the advantages of hybrid RAG approaches?"
    
    try:
        # Get RAG response
        start_time = time.time()
        rag_result = process_query(
            query=query,
            enhancement_method="llm_rewrite",
            strategy_name="hybrid",
            top_k=3,
            return_documents=True
        )
        duration = time.time() - start_time
        
        logger.info(f"RAG processing completed in {duration:.2f}s")
        logger.info(f"Response: {rag_result['response'][:100]}...")
        
        # Extract needed data
        response = rag_result['response']
        documents = [
            Document(
                id=doc['id'],
                content=doc['content'],
                metadata=doc['metadata'],
                score=doc.get('score', 0.0)
            )
            for doc in rag_result.get('documents', [])
        ]
        
        # Evaluate the response
        logger.info("Evaluating RAG response...")
        evaluation_methods = ["relevance", "faithfulness", "answer_relevancy"]
        results = evaluate_response(query, response, documents, methods=evaluation_methods)
        
        for method, result in results.items():
            logger.info(f"{method.capitalize()} score: {result.score:.2f}")
            logger.info(f"Explanation: {result.explanation}")
            logger.info("-" * 30)
        
        # Calculate metrics
        metrics = calculate_metrics(query, response, documents, metrics=["relevance", "faithfulness"])
        logger.info(f"Calculated metrics: {metrics}")
        
        logger.info("-" * 50)
    except Exception as e:
        logger.error(f"Error testing with real RAG output: {str(e)}")


if __name__ == "__main__":
    """Run the test functions."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if test_name == "individual":
            test_individual_metrics()
        elif test_name == "multiple":
            test_multiple_methods()
        elif test_name == "real":
            test_with_real_rag_output()
        else:
            logger.error(f"Unknown test name: {test_name}")
            logger.info("Available tests: individual, multiple, real")
    else:
        # Run all tests
        logger.info("Running all tests...")
        test_individual_metrics()
        test_multiple_methods()
        test_with_real_rag_output()
        logger.info("All tests completed!") 