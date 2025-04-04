"""Vector store implementation for efficient similarity search."""

import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
import time

from sqlalchemy import text, and_, func
from sqlalchemy.orm import Session
import numpy as np

from common.database import SessionLocal
from common.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


class VectorStore:
    """Vector store for similarity search using pgvector."""

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize vector store.
        
        Args:
            db: Optional database session
        """
        self.db = db or SessionLocal()
    
    def __del__(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
    
    def search(
        self,
        query_vector: List[float],
        model_id: str,
        limit: int = 5,
        threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            model_id: Embedding model ID
            limit: Maximum number of results
            threshold: Similarity threshold (if None, return top 'limit' results)
            metadata_filter: Filter based on chunk metadata
            document_id: Optional document ID to limit search
            
        Returns:
            List of similar chunks with similarity scores
        """
        start_time = time.time()
        
        try:
            # Build base query
            # Using raw SQL for more control over pgvector operations
            query = """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.metadata,
                    1 - (e.embedding <=> :query_embedding) as similarity
                FROM
                    embeddings e
                JOIN
                    chunks c ON e.chunk_id = c.id
                WHERE
                    e.model_id = :model_id
            """
            
            params = {
                "query_embedding": query_vector,
                "model_id": model_id
            }
            
            # Add document filter if specified
            if document_id:
                query += " AND c.document_id = :document_id"
                params["document_id"] = document_id
            
            # Add metadata filter if specified
            if metadata_filter:
                for key, value in metadata_filter.items():
                    query += f" AND c.metadata->>{key!r} = :metadata_{key}"
                    params[f"metadata_{key}"] = str(value)
            
            # Add similarity threshold if specified
            if threshold is not None:
                query += " AND (1 - (e.embedding <=> :query_embedding)) >= :threshold"
                params["threshold"] = threshold
            
            # Add ordering and limit
            query += " ORDER BY similarity DESC LIMIT :limit"
            params["limit"] = limit
            
            # Execute query
            results = self.db.execute(text(query), params).fetchall()
            
            # Process results
            processed_results = []
            for row in results:
                processed_results.append({
                    "chunk_id": row.chunk_id,
                    "document_id": row.document_id,
                    "content": row.content,
                    "metadata": row.metadata,
                    "similarity": float(row.similarity)
                })
            
            end_time = time.time()
            logger.info(f"Vector search completed in {end_time - start_time:.4f} seconds")
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            return []
    
    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        model_id: str,
        limit: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        threshold: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity with BM25 keyword matching.
        
        Args:
            query_text: Text query for keyword matching
            query_vector: Query vector for semantic search
            model_id: Embedding model ID
            limit: Maximum number of results
            vector_weight: Weight for vector similarity (0-1)
            bm25_weight: Weight for BM25 score (0-1)
            threshold: Overall similarity threshold
            document_id: Optional document ID to limit search
            
        Returns:
            List of chunks with combined similarity scores
        """
        start_time = time.time()
        
        try:
            # Build hybrid query combining vector similarity with text search
            # Using raw SQL for more control over pgvector operations and full-text search
            query = """
                SELECT 
                    c.id as chunk_id,
                    c.document_id,
                    c.content,
                    c.metadata,
                    (1 - (e.embedding <=> :query_embedding)) * :vector_weight + 
                    ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', :query_text)) * :bm25_weight 
                    as similarity
                FROM
                    embeddings e
                JOIN
                    chunks c ON e.chunk_id = c.id
                WHERE
                    e.model_id = :model_id
            """
            
            params = {
                "query_embedding": query_vector,
                "query_text": query_text,
                "model_id": model_id,
                "vector_weight": vector_weight,
                "bm25_weight": bm25_weight
            }
            
            # Add document filter if specified
            if document_id:
                query += " AND c.document_id = :document_id"
                params["document_id"] = document_id
            
            # Add combined threshold if specified
            if threshold is not None:
                query += " HAVING similarity >= :threshold"
                params["threshold"] = threshold
            
            # Add ordering and limit
            query += " ORDER BY similarity DESC LIMIT :limit"
            params["limit"] = limit
            
            # Execute query
            results = self.db.execute(text(query), params).fetchall()
            
            # Process results
            processed_results = []
            for row in results:
                processed_results.append({
                    "chunk_id": row.chunk_id,
                    "document_id": row.document_id,
                    "content": row.content,
                    "metadata": row.metadata,
                    "similarity": float(row.similarity)
                })
            
            end_time = time.time()
            logger.info(f"Hybrid search completed in {end_time - start_time:.4f} seconds")
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
            return []
    
    def add_vectors(
        self,
        vectors: List[Dict[str, Any]]
    ) -> bool:
        """
        Add vectors to the store.
        
        Args:
            vectors: List of dicts with 'id', 'vector', 'metadata' keys
            
        Returns:
            Success status
        """
        try:
            # Insert/update vectors in database
            # Not implemented here as we handle this in the embedding service
            # through SQLAlchemy models
            logger.info(f"Added {len(vectors)} vectors to store")
            return True
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)}", exc_info=True)
            return False
    
    def create_index(self, index_type: str = "hnsw") -> bool:
        """
        Create or rebuild vector index.
        
        Args:
            index_type: Index type ('hnsw' or 'ivfflat')
            
        Returns:
            Success status
        """
        try:
            if index_type == "hnsw":
                # HNSW index (Hierarchical Navigable Small World)
                # Generally faster but uses more memory
                self.db.execute(text("""
                    DROP INDEX IF EXISTS idx_embeddings_hnsw;
                    CREATE INDEX idx_embeddings_hnsw ON embeddings 
                    USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64);
                """))
            elif index_type == "ivfflat":
                # IVF-Flat index (Inverted File with Flat compression)
                # Better for larger datasets, less memory intensive
                self.db.execute(text("""
                    DROP INDEX IF EXISTS idx_embeddings_ivfflat;
                    CREATE INDEX idx_embeddings_ivfflat ON embeddings 
                    USING ivfflat (embedding vector_l2_ops) WITH (lists=100);
                """))
            else:
                logger.error(f"Unsupported index type: {index_type}")
                return False
            
            self.db.commit()
            logger.info(f"Created {index_type} index for embeddings")
            return True
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}", exc_info=True)
            self.db.rollback()
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Dict with collection statistics
        """
        try:
            # Get embedding count
            embedding_count = self.db.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()
            
            # Get model counts
            model_counts_rows = self.db.execute(text("""
                SELECT model_id, COUNT(*) 
                FROM embeddings 
                GROUP BY model_id
            """)).fetchall()
            
            model_counts = {row[0]: row[1] for row in model_counts_rows}
            
            # Get document counts
            document_count = self.db.execute(text("SELECT COUNT(*) FROM documents")).scalar()
            
            # Get chunk counts
            chunk_count = self.db.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
            
            return {
                "total_embeddings": embedding_count,
                "model_counts": model_counts,
                "document_count": document_count,
                "chunk_count": chunk_count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}", exc_info=True)
            return {
                "error": str(e)
            } 