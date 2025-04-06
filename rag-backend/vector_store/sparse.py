"""Sparse vector retrieval using BM25 and TF-IDF."""

import logging
import math
import uuid
from typing import List, Dict, Any, Optional, Tuple
import time
import re
from collections import Counter

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from common.database import SessionLocal
from common.logging import get_logger
from vector_store.models import Document

# Initialize logger
logger = get_logger(__name__)


class BM25Retriever:
    """BM25 retrieval for keyword-based search."""
    
    def __init__(
        self,
        db: Optional[Session] = None,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            db: Optional database session
            k1: Term saturation parameter
            b: Document length normalization parameter
        """
        self.db = db or SessionLocal()
        self.k1 = k1
        self.b = b
        
        # Cache for collection stats to avoid repeated DB calls
        self._avg_doc_length = None
        self._total_docs = None
        self._idf_cache = {}
    
    def __del__(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def get_term_frequency(self, text: str) -> Dict[str, int]:
        """
        Get term frequency for a document.
        
        Args:
            text: Document text
            
        Returns:
            Dict mapping terms to their frequency
        """
        tokens = self.tokenize(text)
        return Counter(tokens)
    
    def get_idf(self, term: str) -> float:
        """
        Get inverse document frequency for a term.
        
        Args:
            term: Term to get IDF for
            
        Returns:
            IDF value
        """
        # Check cache first
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        try:
            # Count documents containing the term
            query = """
                SELECT COUNT(*) 
                FROM chunks 
                WHERE content ILIKE :term_pattern
            """
            term_pattern = f'%{term}%'
            doc_count = self.db.execute(text(query), {"term_pattern": term_pattern}).scalar()
            
            # Calculate IDF
            total_docs = self.get_total_docs()
            idf = math.log((total_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)
            
            # Cache the result
            self._idf_cache[term] = idf
            
            return idf
        except Exception as e:
            logger.error(f"Error calculating IDF for term '{term}': {str(e)}")
            return 1.0  # Default to neutral IDF
    
    def get_total_docs(self) -> int:
        """
        Get total number of documents in the collection.
        
        Returns:
            Document count
        """
        if self._total_docs is None:
            self._total_docs = self.db.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
        return self._total_docs
    
    def get_avg_doc_length(self) -> float:
        """
        Get average document length in the collection.
        
        Returns:
            Average document length in tokens
        """
        if self._avg_doc_length is None:
            try:
                # Use token count as document length
                query = """
                    SELECT AVG(ARRAY_LENGTH(
                        STRING_TO_ARRAY(
                            REGEXP_REPLACE(LOWER(content), '[^a-z0-9 ]', ' ', 'g'),
                            ' '
                        ), 1
                    ))
                    FROM chunks
                """
                self._avg_doc_length = self.db.execute(text(query)).scalar() or 100.0
            except Exception as e:
                logger.error(f"Error calculating average document length: {str(e)}")
                self._avg_doc_length = 100.0  # Default value
        
        return self._avg_doc_length
    
    def calculate_bm25_score(self, query_terms: List[str], doc_text: str) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: Query terms
            doc_text: Document text
            
        Returns:
            BM25 score
        """
        # Get document stats
        doc_term_freq = self.get_term_frequency(doc_text)
        doc_length = sum(doc_term_freq.values())
        avg_doc_length = self.get_avg_doc_length()
        
        # Calculate BM25 score
        score = 0.0
        for term in query_terms:
            if term in doc_term_freq:
                # Get term frequency in document
                tf = doc_term_freq[term]
                
                # Get IDF for term
                idf = self.get_idf(term)
                
                # Calculate BM25 score component for this term
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                score += idf * (numerator / denominator)
        
        return score
    
    def search(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> List[Document]:
        """
        Search for documents matching query using BM25.
        
        Args:
            query: Search query
            limit: Maximum number of results
            document_id: Optional document ID to filter results
            
        Returns:
            List of Document objects with BM25 scores
        """
        start_time = time.time()
        logger.info(f"BM25 search for query: {query}")
        
        try:
            # Tokenize query
            query_terms = self.tokenize(query)
            
            # First, filter potential matches using a simple LIKE query
            # This avoids calculating BM25 for all documents
            filter_query = """
                SELECT id, document_id, content, metadata
                FROM chunks
                WHERE 1=1
            """
            
            params = {}
            
            # Add document filter if specified
            if document_id:
                filter_query += " AND document_id = :document_id"
                params["document_id"] = document_id
            
            # Add term filters (OR condition for any term)
            if query_terms:
                filter_query += " AND ("
                for i, term in enumerate(query_terms):
                    if i > 0:
                        filter_query += " OR "
                    filter_query += f"content ILIKE :term_{i}"
                    params[f"term_{i}"] = f"%{term}%"
                filter_query += ")"
            
            # Limit initial candidates for more detailed scoring
            filter_query += " LIMIT 100"
            
            # Execute initial filtering
            candidates = self.db.execute(text(filter_query), params).fetchall()
            
            # Calculate BM25 scores for candidates
            scored_results = []
            for row in candidates:
                doc_id = row.id
                doc_text = row.content
                
                # Calculate BM25 score
                score = self.calculate_bm25_score(query_terms, doc_text)
                
                doc = Document(
                    id=doc_id,
                    content=doc_text,
                    metadata=row.metadata or {}
                )
                doc.score = score
                scored_results.append(doc)
            
            # Sort by score and limit
            scored_results.sort(key=lambda x: x.score, reverse=True)
            top_results = scored_results[:limit]
            
            end_time = time.time()
            logger.info(f"BM25 search completed in {end_time - start_time:.4f} seconds")
            
            return top_results
        
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}", exc_info=True)
            return []


class TFIDF:
    """TF-IDF based sparse vector retrieval."""
    
    def __init__(self, db: Optional[Session] = None):
        """
        Initialize TF-IDF retriever.
        
        Args:
            db: Optional database session
        """
        self.db = db or SessionLocal()
        self._idf_cache = {}
    
    def __del__(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def get_idf(self, term: str) -> float:
        """Get inverse document frequency for a term."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        try:
            # Count documents containing the term
            query = """
                SELECT COUNT(*) 
                FROM chunks 
                WHERE content ILIKE :term_pattern
            """
            term_pattern = f'%{term}%'
            doc_count = self.db.execute(text(query), {"term_pattern": term_pattern}).scalar()
            
            # Get total document count
            total_docs = self.db.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
            
            # Calculate IDF
            idf = math.log(total_docs / (1 + doc_count))
            
            # Cache the result
            self._idf_cache[term] = idf
            
            return idf
        except Exception as e:
            logger.error(f"Error calculating IDF for term '{term}': {str(e)}")
            return 1.0
    
    def calculate_tf_idf_vector(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF vector for a document."""
        tokens = self.tokenize(text)
        term_freq = Counter(tokens)
        
        # Calculate TF-IDF for each term
        tfidf_vector = {}
        for term, freq in term_freq.items():
            tf = freq / len(tokens)
            idf = self.get_idf(term)
            tfidf_vector[term] = tf * idf
        
        return tfidf_vector
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two sparse vectors."""
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        # Calculate dot product for common terms
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> List[Document]:
        """Search for documents using TF-IDF vectors."""
        start_time = time.time()
        
        try:
            # Calculate query vector
            query_vector = self.calculate_tf_idf_vector(query)
            
            # Get candidate documents
            filter_query = """
                SELECT id, document_id, content, metadata
                FROM chunks
                WHERE 1=1
            """
            
            params = {}
            
            # Add document filter if specified
            if document_id:
                filter_query += " AND document_id = :document_id"
                params["document_id"] = document_id
            
            # Add term filters
            query_terms = list(query_vector.keys())
            if query_terms:
                filter_query += " AND ("
                for i, term in enumerate(query_terms):
                    if i > 0:
                        filter_query += " OR "
                    filter_query += f"content ILIKE :term_{i}"
                    params[f"term_{i}"] = f"%{term}%"
                filter_query += ")"
            
            # Limit initial candidates
            filter_query += " LIMIT 100"
            
            # Execute initial filtering
            candidates = self.db.execute(text(filter_query), params).fetchall()
            
            # Calculate TF-IDF scores for candidates
            scored_results = []
            for row in candidates:
                doc_id = row.id
                doc_text = row.content
                
                # Calculate TF-IDF vector for document
                doc_vector = self.calculate_tf_idf_vector(doc_text)
                
                # Calculate cosine similarity with query vector
                similarity = self.cosine_similarity(query_vector, doc_vector)
                
                doc = Document(
                    id=doc_id,
                    content=doc_text,
                    metadata=row.metadata or {}
                )
                doc.score = similarity
                scored_results.append(doc)
            
            # Sort by similarity and limit
            scored_results.sort(key=lambda x: x.score, reverse=True)
            top_results = scored_results[:limit]
            
            end_time = time.time()
            logger.info(f"TF-IDF search completed in {end_time - start_time:.4f} seconds")
            
            return top_results
        
        except Exception as e:
            logger.error(f"Error in TF-IDF search: {str(e)}", exc_info=True)
            return []


# Module-level functions for API compatibility


def search_bm25(query: str, limit: int = 5, document_id: Optional[str] = None) -> List[Document]:
    """
    Search for documents using BM25 scoring.
    
    Args:
        query: Search query
        limit: Maximum number of results
        document_id: Optional document ID to filter results
        
    Returns:
        List of Document objects
    """
    retriever = BM25Retriever()
    try:
        results = retriever.search(query, limit, document_id)
        return results
    finally:
        del retriever  # Ensure resources are cleaned up


def search_tfidf(query: str, limit: int = 5, document_id: Optional[str] = None) -> List[Document]:
    """
    Search for documents using TF-IDF scoring.
    
    Args:
        query: Search query
        limit: Maximum number of results
        document_id: Optional document ID to filter results
        
    Returns:
        List of Document objects
    """
    retriever = TFIDF()
    try:
        results = retriever.search(query, limit, document_id)
        return results
    finally:
        del retriever  # Ensure resources are cleaned up


def keyword_search(
    query: str,
    algorithm: str = "bm25",
    limit: int = 5,
    document_id: Optional[str] = None
) -> List[Document]:
    """
    Search for documents using keyword-based retrieval.
    
    Args:
        query: Search query
        algorithm: Algorithm to use ('bm25' or 'tfidf')
        limit: Maximum number of results
        document_id: Optional document ID to filter results
        
    Returns:
        List of Document objects
    """
    if algorithm.lower() == "bm25":
        return search_bm25(query, limit, document_id)
    elif algorithm.lower() == "tfidf":
        return search_tfidf(query, limit, document_id)
    else:
        logger.error(f"Unsupported algorithm: {algorithm}")
        return [] 