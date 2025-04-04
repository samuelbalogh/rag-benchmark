"""Retrieval strategies for the RAG benchmark platform."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union

import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

from vector_store.service import get_top_documents, get_documents_by_metadata, hybrid_search
from vector_store.models import Document
from embedding_service.service import get_embedding
from knowledge_graph.service import get_entity_subgraph, get_related_nodes, search_graph
from common.logging import get_logger

# Set up logging
logger = get_logger(__name__)


class RagStrategy:
    """Base class for RAG retrieval strategies."""
    
    def __init__(self, **kwargs):
        """Initialize the strategy with optional parameters."""
        self.parameters = kwargs
    
    def retrieve(
        self, 
        query: str, 
        enhanced_query: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve documents relevant to the query.
        
        Args:
            query: Original query string
            enhanced_query: Enhanced version of the query (optional)
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Tuple of (list of retrieved documents, metadata about the retrieval process)
        """
        raise NotImplementedError("Subclasses must implement retrieve method")


class VectorSearchStrategy(RagStrategy):
    """Strategy for vector-based document retrieval."""
    
    def retrieve(
        self, 
        query: str, 
        enhanced_query: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve documents using vector search.
        
        Args:
            query: Original query string
            enhanced_query: Enhanced version of the query (optional)
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Tuple of (list of retrieved documents, metadata about the retrieval process)
        """
        start_time = time.time()
        metadata = {"strategy": "vector_search", "parameters": {**self.parameters, **kwargs}}
        
        # Use enhanced query if available, otherwise use original query
        search_query = enhanced_query if enhanced_query and isinstance(enhanced_query, str) else query
        
        try:
            # Get query embedding
            query_embedding = get_embedding(search_query)
            
            # Get top documents
            documents = get_top_documents(
                query_embedding, 
                top_k=top_k, 
                filters=kwargs.get("filters", None)
            )
            
            # Handle multiple enhanced queries (if enhanced_query is a list)
            if enhanced_query and isinstance(enhanced_query, list):
                all_docs = []
                for subquery in enhanced_query:
                    subquery_embedding = get_embedding(subquery)
                    subquery_docs = get_top_documents(
                        subquery_embedding, 
                        top_k=max(2, top_k // len(enhanced_query)),
                        filters=kwargs.get("filters", None)
                    )
                    all_docs.extend(subquery_docs)
                
                # Deduplicate and take top_k
                unique_docs = {}
                for doc in all_docs:
                    if doc.id not in unique_docs:
                        unique_docs[doc.id] = doc
                
                documents = list(unique_docs.values())[:top_k]
                metadata["enhanced_queries_used"] = len(enhanced_query)
            
            # Record metadata
            metadata["time_taken"] = time.time() - start_time
            metadata["num_documents"] = len(documents)
            
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Error in vector search strategy: {str(e)}")
            metadata["error"] = str(e)
            return [], metadata


class KnowledgeGraphStrategy(RagStrategy):
    """Strategy for knowledge graph-based document retrieval."""
    
    def retrieve(
        self, 
        query: str, 
        enhanced_query: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve documents using knowledge graph.
        
        Args:
            query: Original query string
            enhanced_query: Enhanced version of the query (optional)
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Tuple of (list of retrieved documents, metadata about the retrieval process)
        """
        start_time = time.time()
        metadata = {"strategy": "knowledge_graph", "parameters": {**self.parameters, **kwargs}}
        
        try:
            # Extract entities from query
            query_text = enhanced_query if enhanced_query and isinstance(enhanced_query, str) else query
            
            # Get related entities from knowledge graph
            graph_results = search_graph(query_text)
            
            if not graph_results or not graph_results.get("entities"):
                logger.info(f"No entities found in query: {query_text}")
                return [], metadata
            
            # Get relevant documents based on entity IDs
            entity_ids = [entity["id"] for entity in graph_results.get("entities", [])]
            document_ids = []
            
            # For each entity, get related document IDs
            for entity_id in entity_ids:
                related_nodes = get_related_nodes(entity_id, "Document")
                document_ids.extend([node["id"] for node in related_nodes])
            
            # Remove duplicates
            document_ids = list(set(document_ids))
            
            # Fetch documents by ID
            documents = []
            if document_ids:
                for doc_id in document_ids[:top_k]:
                    docs = get_documents_by_metadata({"id": doc_id})
                    documents.extend(docs)
            
            # If we don't have enough documents, fall back to vector search
            if len(documents) < top_k:
                logger.info(f"Knowledge graph returned only {len(documents)} documents, falling back to vector search")
                # Get query embedding
                query_embedding = get_embedding(query_text)
                
                # Get additional documents
                additional_docs = get_top_documents(
                    query_embedding, 
                    top_k=top_k - len(documents),
                    filters=kwargs.get("filters", None)
                )
                
                # Add to results, ensuring no duplicates
                existing_ids = {doc.id for doc in documents}
                for doc in additional_docs:
                    if doc.id not in existing_ids:
                        documents.append(doc)
                        existing_ids.add(doc.id)
            
            # Record metadata
            metadata["time_taken"] = time.time() - start_time
            metadata["num_documents"] = len(documents)
            metadata["entities_found"] = len(entity_ids)
            
            return documents[:top_k], metadata
            
        except Exception as e:
            logger.error(f"Error in knowledge graph strategy: {str(e)}")
            metadata["error"] = str(e)
            return [], metadata


class HybridStrategy(RagStrategy):
    """Strategy combining vector search and knowledge graph retrieval."""
    
    def retrieve(
        self, 
        query: str, 
        enhanced_query: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Retrieve documents using both vector search and knowledge graph.
        
        Args:
            query: Original query string
            enhanced_query: Enhanced version of the query (optional)
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Tuple of (list of retrieved documents, metadata about the retrieval process)
        """
        start_time = time.time()
        metadata = {"strategy": "hybrid", "parameters": {**self.parameters, **kwargs}}
        
        try:
            # Run both strategies in parallel
            query_text = enhanced_query if enhanced_query and isinstance(enhanced_query, str) else query
            
            # Get query embedding
            query_embedding = get_embedding(query_text)
            
            # Get documents using hybrid search
            hybrid_weight = kwargs.get("hybrid_weight", 0.7)  # Default weight for vector vs semantic
            documents = hybrid_search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                vector_weight=hybrid_weight,
                semantic_weight=1.0 - hybrid_weight,
                filters=kwargs.get("filters", None)
            )
            
            # Record metadata
            metadata["time_taken"] = time.time() - start_time
            metadata["num_documents"] = len(documents)
            metadata["hybrid_weight"] = hybrid_weight
            
            return documents, metadata
            
        except Exception as e:
            logger.error(f"Error in hybrid strategy: {str(e)}")
            metadata["error"] = str(e)
            return [], metadata


# Factory method to create strategy instances
def get_strategy(strategy_name: str, **kwargs) -> RagStrategy:
    """Get a retrieval strategy by name.
    
    Args:
        strategy_name: Name of the strategy to use
        **kwargs: Parameters to pass to the strategy
        
    Returns:
        RagStrategy instance
    """
    strategies = {
        "vector_search": VectorSearchStrategy,
        "knowledge_graph": KnowledgeGraphStrategy,
        "hybrid": HybridStrategy
    }
    
    strategy_class = strategies.get(strategy_name)
    if not strategy_class:
        logger.warning(f"Unknown strategy: {strategy_name}, falling back to vector search")
        strategy_class = VectorSearchStrategy
    
    return strategy_class(**kwargs)
