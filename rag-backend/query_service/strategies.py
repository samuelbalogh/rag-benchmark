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
from knowledge_graph_service.service import get_graph, extract_entities, extract_relevant_subgraph
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
            
            # Extract document_id if specified
            document_id = kwargs.get("document_id") or self.parameters.get("document_id")
            model_id = kwargs.get("model_id") or self.parameters.get("model_id", "text-embedding-ada-002")
            
            # Get top documents
            documents = get_top_documents(
                query_vector=query_embedding, 
                k=top_k,
                model_id=model_id,
                filters=kwargs.get("filters"),
                document_id=document_id
            )
            
            # Handle multiple enhanced queries (if enhanced_query is a list)
            if enhanced_query and isinstance(enhanced_query, list):
                all_docs = []
                for subquery in enhanced_query:
                    subquery_embedding = get_embedding(subquery)
                    subquery_docs = get_top_documents(
                        query_vector=subquery_embedding, 
                        k=max(2, top_k // len(enhanced_query)),
                        model_id=model_id,
                        filters=kwargs.get("filters"),
                        document_id=document_id
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
            
            # Get document_id from parameters
            document_id = kwargs.get("document_id")
            if not document_id:
                logger.error("KnowledgeGraphStrategy requires document_id parameter")
                return [], metadata
                
            # Get graph for the document
            graph = get_graph(document_id)
            if not graph:
                logger.warning(f"No knowledge graph found for document: {document_id}")
                return [], metadata
                
            # Extract entities from query
            query_entities = extract_entities(query_text)
            
            # Find relevant subgraph
            subgraph = extract_relevant_subgraph(graph, query_entities)
            
            # If no relevant subgraph found, fall back to vector search
            if not subgraph or not subgraph.nodes():
                logger.info(f"No relevant subgraph found for query: {query_text}")
                
                # Fall back to vector search
                query_embedding = get_embedding(query_text)
                
                documents = get_top_documents(
                    query_embedding, 
                    top_k=top_k,
                    filters=kwargs.get("filters", None)
                )
                
                metadata["fallback_to_vector"] = True
                metadata["time_taken"] = time.time() - start_time
                metadata["num_documents"] = len(documents)
                
                return documents, metadata
            
            # Get document chunks based on the subgraph
            # In a real implementation, we would link subgraph nodes to document chunks
            # Here we're using a simplified approach
            
            # Simulate fetching documents related to entities in the subgraph
            entity_names = list(subgraph.nodes())
            
            # Get documents by metadata 
            # This assumes document metadata contains entity information
            documents = []
            for entity in entity_names:
                entity_docs = get_documents_by_metadata({"entities": entity})
                documents.extend(entity_docs)
            
            # Deduplicate documents
            unique_docs = {}
            for doc in documents:
                if doc.id not in unique_docs:
                    unique_docs[doc.id] = doc
            
            documents = list(unique_docs.values())[:top_k]
            
            # If we don't have enough documents, fall back to vector search
            if len(documents) < top_k:
                logger.info(f"Knowledge graph returned only {len(documents)} documents, adding vector search results")
                query_embedding = get_embedding(query_text)
                
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
            metadata["entities_found"] = len(entity_names)
            
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
            # Get the query text to use
            query_text = enhanced_query if enhanced_query and isinstance(enhanced_query, str) else query
            
            # Get query embedding
            query_embedding = get_embedding(query_text)
            
            # Extract parameters
            document_id = kwargs.get("document_id") or self.parameters.get("document_id")
            model_id = kwargs.get("model_id") or self.parameters.get("model_id", "text-embedding-ada-002")
            vector_weight = kwargs.get("vector_weight") or self.parameters.get("vector_weight", 0.7)
            bm25_weight = kwargs.get("bm25_weight") or self.parameters.get("bm25_weight", 0.3)
            metadata_filters = kwargs.get("metadata") or self.parameters.get("metadata", {})
            
            # Use hybrid search
            documents = hybrid_search(
                query_vector=query_embedding,
                metadata=metadata_filters,
                k=top_k,
                query_text=query_text,
                model_id=model_id,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                document_id=document_id
            )
            
            # Record metadata
            metadata["time_taken"] = time.time() - start_time
            metadata["num_documents"] = len(documents)
            metadata["hybrid_weight"] = vector_weight
            
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
