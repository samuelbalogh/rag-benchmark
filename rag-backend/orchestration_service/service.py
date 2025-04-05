"""RAG orchestration service for coordinating RAG pipeline components."""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union

from query_service.service import process_query
from query_service.query_enhancement import enhance_query, combine_enhancement_methods
from query_service.strategies import get_strategy
from evaluation_service.service import evaluate_response, calculate_metrics
from common.logging import get_logger
from common.config import get_settings

# Initialize logger
logger = get_logger(__name__)


class RagOrchestrator:
    """Orchestrates the RAG pipeline components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator with optional configuration.
        
        Args:
            config: Configuration for the orchestrator
        """
        self.config = config or {}
        self.settings = get_settings()
        
        # Default configurations for different RAG strategies
        self.default_configs = {
            "vector": {
                "enhancement_method": "llm_rewrite",
                "strategy_name": "vector_search",
                "top_k": 5,
                "llm_model": "gpt-3.5-turbo"
            },
            "knowledge_graph": {
                "enhancement_method": None,
                "strategy_name": "knowledge_graph",
                "top_k": 5,
                "llm_model": "gpt-3.5-turbo"
            },
            "hybrid": {
                "enhancement_method": "query_decomposition",
                "strategy_name": "hybrid",
                "strategy_params": {"hybrid_weight": 0.7},
                "top_k": 5,
                "llm_model": "gpt-3.5-turbo"
            },
            "enhanced_vector": {
                "enhancement_method": "combined",
                "enhancement_methods": ["llm_rewrite", "synonym_expansion"],
                "strategy_name": "vector_search",
                "top_k": 8,
                "llm_model": "gpt-3.5-turbo"
            }
        }
    
    def process_with_strategy(
        self, 
        query: str, 
        strategy_key: str,
        document_ids: Optional[List[str]] = None,
        override_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a query using a specific predefined strategy.
        
        Args:
            query: Query to process
            strategy_key: Key for the strategy configuration
            document_ids: Optional list of document IDs to search
            override_params: Optional parameters to override default configuration
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        # Get strategy configuration
        if strategy_key not in self.default_configs:
            logger.warning(f"Unknown strategy key: {strategy_key}, defaulting to vector")
            strategy_key = "vector"
        
        # Merge configurations
        config = self.default_configs[strategy_key].copy()
        if override_params:
            config.update(override_params)
        
        # Add document filters if document_ids are provided
        if document_ids:
            config["filters"] = {"document_id": document_ids}
        
        # Process the query
        logger.info(f"Processing query with strategy: {strategy_key}")
        result = process_query(
            query=query,
            **config,
            **kwargs
        )
        
        # Add strategy info to result
        result["strategy_key"] = strategy_key
        
        return result
    
    def compare_strategies(
        self, 
        query: str, 
        strategy_keys: List[str],
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a query using multiple strategies and compare results.
        
        Args:
            query: Query to process
            strategy_keys: List of strategy keys to compare
            document_ids: Optional list of document IDs to search
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with comparison results
        """
        start_time = time.time()
        
        # Process query with each strategy
        results = {}
        for strategy_key in strategy_keys:
            try:
                strategy_result = self.process_with_strategy(
                    query=query,
                    strategy_key=strategy_key,
                    document_ids=document_ids,
                    **kwargs
                )
                results[strategy_key] = strategy_result
            except Exception as e:
                logger.error(f"Error processing query with strategy {strategy_key}: {str(e)}")
                results[strategy_key] = {"error": str(e)}
        
        # Compare metrics
        comparison = {
            "query": query,
            "strategies": strategy_keys,
            "results": results,
            "metrics_comparison": self._compare_metrics(results),
            "processing_time": time.time() - start_time
        }
        
        return comparison
    
    def _compare_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare metrics across different strategies.
        
        Args:
            results: Dictionary of results from different strategies
            
        Returns:
            Dictionary with metric comparisons
        """
        # Common metrics to compare
        metrics = ["response_time", "length"]
        
        # Extract metrics from each result
        metrics_by_strategy = {}
        for strategy_key, result in results.items():
            if "metrics" in result:
                metrics_by_strategy[strategy_key] = result["metrics"]
        
        # Calculate comparative metrics
        comparison = {}
        for metric in metrics:
            values = {k: v.get(metric, 0) for k, v in metrics_by_strategy.items() if metric in v}
            if values:
                comparison[metric] = {
                    "values": values,
                    "min": min(values.items(), key=lambda x: x[1]),
                    "max": max(values.items(), key=lambda x: x[1]),
                    "avg": sum(values.values()) / len(values)
                }
        
        return comparison
    
    def process_complex_query(
        self, 
        query: str,
        document_ids: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process a complex query using an adaptive approach.
        
        This method analyzes the query and selects the most appropriate
        strategy based on the query characteristics.
        
        Args:
            query: Complex query to process
            document_ids: Optional list of document IDs to search
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Step 1: Analyze the query to determine the most appropriate strategy
        query_analysis = self._analyze_query(query)
        
        # Step 2: Select strategy based on query analysis
        selected_strategy = query_analysis["recommended_strategy"]
        logger.info(f"Selected strategy for complex query: {selected_strategy}")
        
        # Step 3: Process query with selected strategy
        result = self.process_with_strategy(
            query=query,
            strategy_key=selected_strategy,
            document_ids=document_ids,
            **kwargs
        )
        
        # Add query analysis to result
        result["query_analysis"] = query_analysis
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query to determine its characteristics.
        
        Args:
            query: Query to analyze
            
        Returns:
            Dictionary with query analysis
        """
        try:
            # Determine complexity and type of query
            query_length = len(query.split())
            has_multiple_questions = "?" in query and query.count("?") > 1
            
            # Check for entity-focused queries
            entity_indicators = ["who", "where", "when", "which", "whose"]
            is_entity_focused = any(query.lower().startswith(word) for word in entity_indicators)
            
            # Check for complex relationship queries
            relationship_indicators = ["relationship", "related", "connection", "between", "compare"]
            is_relationship_query = any(word in query.lower() for word in relationship_indicators)
            
            # Check for descriptive queries
            descriptive_indicators = ["describe", "explain", "what is", "how does", "why"]
            is_descriptive = any(phrase in query.lower() for phrase in descriptive_indicators)
            
            # Simple categorization
            if is_relationship_query or (is_entity_focused and query_length > 10):
                recommended_strategy = "knowledge_graph"
            elif has_multiple_questions or query_length > 15:
                recommended_strategy = "hybrid"
            elif is_descriptive and query_length > 8:
                recommended_strategy = "enhanced_vector"
            else:
                recommended_strategy = "vector"
            
            return {
                "query_length": query_length,
                "has_multiple_questions": has_multiple_questions,
                "is_entity_focused": is_entity_focused,
                "is_relationship_query": is_relationship_query,
                "is_descriptive": is_descriptive,
                "recommended_strategy": recommended_strategy
            }
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return {"recommended_strategy": "vector"}


# Create a singleton instance
orchestrator = RagOrchestrator()


def process_query_with_strategy(
    query: str, 
    strategy_key: str = "vector",
    document_ids: Optional[List[str]] = None,
    override_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Process a query using a specific strategy.
    
    Args:
        query: Query to process
        strategy_key: Key for the strategy configuration
        document_ids: Optional list of document IDs to search
        override_params: Optional parameters to override default configuration
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results
    """
    return orchestrator.process_with_strategy(
        query=query,
        strategy_key=strategy_key,
        document_ids=document_ids,
        override_params=override_params,
        **kwargs
    )


def compare_strategies(
    query: str, 
    strategy_keys: List[str] = ["vector", "knowledge_graph", "hybrid"],
    document_ids: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Compare multiple strategies for a query.
    
    Args:
        query: Query to process
        strategy_keys: List of strategy keys to compare
        document_ids: Optional list of document IDs to search
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with comparison results
    """
    return orchestrator.compare_strategies(
        query=query,
        strategy_keys=strategy_keys,
        document_ids=document_ids,
        **kwargs
    )


def process_complex_query(
    query: str,
    document_ids: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Process a complex query using an adaptive approach.
    
    Args:
        query: Complex query to process
        document_ids: Optional list of document IDs to search
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results
    """
    return orchestrator.process_complex_query(
        query=query,
        document_ids=document_ids,
        **kwargs
    ) 