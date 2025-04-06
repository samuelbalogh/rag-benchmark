"""Query processing service for the RAG benchmark platform."""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from vector_store.models import Document
from query_service.query_enhancement import enhance_query, combine_enhancement_methods
from query_service.strategies import get_strategy
from common.logging import get_logger
from common.db import get_connection
from common.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=get_settings().openai_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def generate_response(query: str, context: List[Document], model: str = "gpt-4o-mini") -> str:
    """Generate a response to the query using the provided context.
    
    Args:
        query: The query to answer
        context: List of context documents
        model: LLM model to use
        
    Returns:
        Generated response
    """
    try:
        # Combine context documents
        context_text = "\n\n".join([f"Document {i+1}:\n{doc.content}" for i, doc in enumerate(context)])
        
        # Create prompt
        prompt = f"""
        Answer the following question based on the context provided. If the answer cannot be determined from the context, 
        say "I don't have enough information to answer that question."
        
        Question: {query}
        
        Context:
        {context_text}
        """
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that provides accurate answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "An error occurred while generating the response."


def evaluate_response(query: str, response: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
    """Evaluate the quality of the response.
    
    Args:
        query: The original query
        response: The generated response
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "length": len(response),
        "response_time": time.time()  # Will be updated with actual duration in process_query
    }
    
    # Add more sophisticated metrics if ground truth is available
    if ground_truth:
        # Simple overlap score for demonstration
        overlap_words = set(response.lower().split()) & set(ground_truth.lower().split())
        metrics["overlap_score"] = len(overlap_words) / max(len(set(ground_truth.lower().split())), 1)
    
    return metrics


def save_query_log(
    query: str,
    enhanced_query: Union[str, List[str]],
    strategy: str,
    documents: List[Document],
    response: str,
    metrics: Dict[str, float],
    metadata: Dict[str, Any]
) -> str:
    """Save query and response to the database for future analysis.
    
    Args:
        query: Original query
        enhanced_query: Enhanced query or queries
        strategy: Strategy used for retrieval
        documents: Retrieved documents
        response: Generated response
        metrics: Evaluation metrics
        metadata: Additional metadata
        
    Returns:
        ID of the saved log entry
    """
    try:
        query_id = str(uuid.uuid4())
        
        # Prepare the log entry
        log_entry = {
            "id": query_id,
            "timestamp": time.time(),
            "original_query": query,
            "enhanced_query": enhanced_query if isinstance(enhanced_query, str) else json.dumps(enhanced_query),
            "strategy": strategy,
            "documents": json.dumps([{
                "id": doc.id,
                "title": doc.metadata.get("title", ""),
                "score": doc.score if hasattr(doc, "score") else None,
                "chunk_id": doc.metadata.get("chunk_id", "")
            } for doc in documents]),
            "response": response,
            "metrics": json.dumps(metrics),
            "metadata": json.dumps(metadata)
        }
        
        # Save to database
        with get_connection() as conn:
            with conn.cursor() as cursor:
                fields = ", ".join(log_entry.keys())
                placeholders = ", ".join(["%s"] * len(log_entry))
                values = list(log_entry.values())
                
                cursor.execute(
                    f"INSERT INTO query_logs ({fields}) VALUES ({placeholders}) RETURNING id",
                    values
                )
                result = cursor.fetchone()
                conn.commit()
                
                if result:
                    query_id = result[0]
        
        logger.info(f"Saved query log with ID: {query_id}")
        return query_id
        
    except Exception as e:
        logger.error(f"Error saving query log: {str(e)}")
        return query_id


def process_query(
    query: str,
    document_ids: List[str] = None,
    strategy_name: str = "vector_search",
    parameters: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Process a query using the specified retrieval strategy.
    
    Args:
        query: The query to process
        document_ids: Optional list of document IDs to search within
        strategy_name: Name of the retrieval strategy to use
        parameters: Parameters for the strategy and processing
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    if parameters is None:
        parameters = {}
    
    # Extract parameters with defaults
    enhancement_method = parameters.get("enhancement_method")
    top_k = parameters.get("top_k", 5)
    llm_model = parameters.get("llm_model", "gpt-4o-mini")
    ground_truth = parameters.get("ground_truth")
    return_documents = parameters.get("return_documents", False)
    
    try:
        # Step 1: Enhance the query if needed
        enhancement_metadata = {}
        
        # Always call enhance_query, it will return the original query if no enhancement method is specified
        enhanced_query = enhance_query(query, enhancement_method, parameters.get("enhancement_params"))
        if enhancement_method:
            logger.info(f"Enhancing query using method: {enhancement_method}")
            
            if enhancement_method == "combined":
                # Use multiple enhancement methods
                methods = parameters.get("enhancement_methods", ["llm_rewrite", "query_decomposition"])
                enhanced_results = combine_enhancement_methods(query, methods)
                enhanced_query = enhanced_results
                enhancement_metadata = {"methods": methods, "results": enhanced_results}
            else:
                # Single enhancement method metadata
                enhancement_metadata = {"method": enhancement_method}
        
        # Step 2: Get the appropriate strategy
        logger.info(f"Using strategy: {strategy_name}")
        strategy_params = parameters.copy()
        if document_ids:
            strategy_params["document_ids"] = document_ids
        strategy = get_strategy(strategy_name, **strategy_params)
        
        # Step 3: Retrieve documents
        documents, retrieval_metadata = strategy.retrieve(
            query=query,
            enhanced_query=enhanced_query,
            top_k=top_k,
            **kwargs
        )
        
        if not documents:
            logger.warning("No documents retrieved")
            return {
                "query": query,
                "enhanced_query": enhanced_query,
                "response": "I couldn't find any relevant information to answer your question.",
                "context": [],
                "metadata": {
                    "strategy": strategy_name,
                    "enhancement": enhancement_metadata,
                    "retrieval": retrieval_metadata,
                    "processing_time": time.time() - start_time
                }
            }
        
        # Step 4: Generate response
        logger.info(f"Generating response with {len(documents)} documents using model: {llm_model}")
        response = generate_response(query, documents, llm_model)
        
        # Step 5: Evaluate response
        metrics = evaluate_response(query, response, ground_truth)
        metrics["response_time"] = time.time() - start_time
        
        # Step 6: Save query log
        log_id = save_query_log(
            query=query,
            enhanced_query=enhanced_query,
            strategy=strategy_name,
            documents=documents,
            response=response,
            metrics=metrics,
            metadata={
                "enhancement": enhancement_metadata,
                "retrieval": retrieval_metadata,
                "model": llm_model
            }
        )
        
        # Step 7: Prepare and return result
        result = {
            "query": query,
            "enhanced_query": enhanced_query,
            "result": response,
            "metrics": metrics,
            "log_id": log_id,
            "metadata": {
                "strategy": strategy_name,
                "enhancement": enhancement_metadata,
                "retrieval": retrieval_metadata,
                "processing_time": time.time() - start_time
            }
        }
        
        if return_documents:
            result["documents"] = [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": doc.score if hasattr(doc, "score") else None
                }
                for doc in documents
            ]
        
        # Add context for compatibility with test
        result["context"] = [doc.content for doc in documents]
        
        logger.info(f"Query processed successfully in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "query": query,
            "error": str(e),
            "metadata": {
                "processing_time": time.time() - start_time
            }
        }


def calculate_metrics(
    query: str, 
    documents: List[Document], 
    response: str, 
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """Calculate evaluation metrics for a response.
    
    Args:
        query: The original query
        documents: The retrieved documents
        response: The generated response
        ground_truth: Optional ground truth answer
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "relevance": 0.0,
        "completeness": 0.0,
        "length": len(response)
    }
    
    # Basic relevance calculation (simplified for testing)
    # Check if response contains terms from the query
    query_terms = set(query.lower().split())
    response_terms = set(response.lower().split())
    query_term_overlap = len(query_terms.intersection(response_terms)) / len(query_terms) if query_terms else 0
    metrics["relevance"] = min(query_term_overlap * 1.5, 1.0)  # Scale up a bit, max 1.0
    
    # Basic completeness calculation
    # Check if the response addresses key points from retrieved documents
    if documents:
        doc_terms = set()
        for doc in documents:
            doc_terms.update(doc.content.lower().split())
        
        doc_term_overlap = len(doc_terms.intersection(response_terms)) / min(100, len(doc_terms)) if doc_terms else 0
        metrics["completeness"] = min(doc_term_overlap * 2.0, 1.0)  # Scale up, max 1.0
    
    # Add ground truth evaluation if available
    if ground_truth:
        truth_terms = set(ground_truth.lower().split())
        truth_term_overlap = len(truth_terms.intersection(response_terms)) / len(truth_terms) if truth_terms else 0
        metrics["accuracy"] = min(truth_term_overlap * 1.5, 1.0)  # Scale up, max 1.0
    
    return metrics 