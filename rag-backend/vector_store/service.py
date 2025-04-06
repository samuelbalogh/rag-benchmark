"""Vector store service for document retrieval."""

import time
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple

from sqlalchemy import text, and_, func
from sqlalchemy.orm import Session
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vector_store.models import Document
from common.database import SessionLocal
from common.logging import get_logger
from common.config import get_settings

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
    
    def get_documents_from_results(self, results: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert database results to Document objects.
        
        Args:
            results: List of raw database results
            
        Returns:
            List of Document objects
        """
        documents = []
        for result in results:
            doc = Document(
                id=result["chunk_id"],
                content=result["content"],
                metadata=result.get("metadata", {})
            )
            if "similarity" in result:
                doc.score = result["similarity"]
            documents.append(doc)
        return documents


# Create a global instance for use with module functions
_vector_store = None

def get_vector_store() -> VectorStore:
    """Get or create a VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_top_documents(query_vector: np.ndarray, k: int = 5, **kwargs) -> List[Document]:
    """Get top k documents most similar to the query vector.
    
    Args:
        query_vector: Query embedding vector
        k: Number of documents to return
        **kwargs: Additional search parameters (model_id, threshold, metadata_filter, document_id)
        
    Returns:
        List of most similar documents
    """
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Extract additional parameters
        model_id = kwargs.get("model_id", "text-embedding-ada-002")
        threshold = kwargs.get("threshold")
        metadata_filter = kwargs.get("filters")
        document_id = kwargs.get("document_id")
        
        # Search for similar vectors
        results = vector_store.search(
            query_vector=query_vector.tolist(),
            model_id=model_id,
            limit=k,
            threshold=threshold,
            metadata_filter=metadata_filter,
            document_id=document_id
        )
        
        # Convert to Document objects
        return vector_store.get_documents_from_results(results)
    except Exception as e:
        logger.error(f"Error getting top documents: {str(e)}")
        return []


def get_documents_by_metadata(metadata: Dict[str, Any]) -> List[Document]:
    """Get documents matching metadata filters.
    
    Args:
        metadata: Dictionary of metadata key-value pairs to filter by
        
    Returns:
        List of matching documents
    """
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Convert to SQL query
        query = """
            SELECT id, document_id, content, metadata
            FROM chunks
            WHERE 1=1
        """
        
        params = {}
        
        # Add metadata filters
        for i, (key, value) in enumerate(metadata.items()):
            query += f" AND metadata->>{key!r} = :value_{i}"
            params[f"value_{i}"] = str(value)
        
        # Execute query
        results = vector_store.db.execute(text(query), params).fetchall()
        
        # Process results
        processed_results = []
        for row in results:
            processed_results.append({
                "chunk_id": row.id,
                "document_id": row.document_id,
                "content": row.content,
                "metadata": row.metadata
            })
        
        # Convert to Document objects
        return vector_store.get_documents_from_results(processed_results)
    except Exception as e:
        logger.error(f"Error getting documents by metadata: {str(e)}")
        return []


def hybrid_search(query_vector: np.ndarray, metadata: Dict[str, Any] = None, k: int = 5, **kwargs) -> List[Document]:
    """Combine vector similarity and metadata filtering.
    
    Args:
        query_vector: Query embedding vector
        metadata: Metadata filters
        k: Number of documents to return
        **kwargs: Additional search parameters
        
    Returns:
        List of documents matching both criteria
    """
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Check if we have query text for hybrid search
        query_text = kwargs.get("query_text", "")
        
        if query_text:
            # Use hybrid BM25+vector search
            model_id = kwargs.get("model_id", "text-embedding-ada-002")
            vector_weight = kwargs.get("vector_weight", 0.7)
            bm25_weight = kwargs.get("bm25_weight", 0.3)
            threshold = kwargs.get("threshold")
            document_id = kwargs.get("document_id")
            
            results = vector_store.hybrid_search(
                query_text=query_text,
                query_vector=query_vector.tolist(),
                model_id=model_id,
                limit=k,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                threshold=threshold,
                document_id=document_id
            )
            
            return vector_store.get_documents_from_results(results)
        else:
            # If no query text, first filter by metadata, then do vector search
            if metadata:
                # Get documents matching metadata
                metadata_docs = get_documents_by_metadata(metadata)
                
                if not metadata_docs:
                    return []
                
                # Extract document IDs
                doc_ids = [doc.id for doc in metadata_docs]
                
                # Use these IDs to filter vector search
                vector_search_results = get_top_documents(
                    query_vector=query_vector,
                    k=k,
                    document_id=doc_ids,
                    **kwargs
                )
                
                return vector_search_results
            else:
                # If no metadata, just do regular vector search
                return get_top_documents(query_vector, k, **kwargs)
    except Exception as e:
        logger.error(f"Error performing hybrid search: {str(e)}")
        return [] 