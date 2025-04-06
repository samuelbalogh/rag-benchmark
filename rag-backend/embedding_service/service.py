"""Embedding service for generating vector embeddings."""

import logging
import os
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json

from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np

from common.database import SessionLocal
from common.logging import get_logger
from common.models import Document, Chunk, Embedding, ProcessingStatus
from embedding_service.models import EmbeddingModel, get_embedding_model

# Initialize logger
logger = get_logger(__name__)


async def generate_embeddings(document_id: str, model_name: str) -> Dict[str, Any]:
    """
    Generate embeddings for document chunks.
    
    Args:
        document_id: ID of the document
        model_name: Name of the embedding model to use
        
    Returns:
        Dictionary with status and embedding information
    """
    logger.info(f"Generating embeddings for document: {document_id} with model: {model_name}")
    
    start_time = time.time()
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Update processing status
        update_document_status(db, document_id, "embedding", "processing")
        
        # Get chunks
        chunks = get_chunks(db, document_id)
        if not chunks:
            logger.warning(f"No chunks found for document: {document_id}")
            update_document_status(db, document_id, "embedding", "failed", "No chunks found")
            return {"success": False, "error": "No chunks found"}
        
        # Get embedding model
        model = get_embedding_model(model_name)
        
        # Get embedding texts
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [str(chunk.id) for chunk in chunks]
        
        # Generate embeddings
        try:
            embeddings = model.embed(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            update_document_status(db, document_id, "embedding", "failed", str(e))
            return {"success": False, "error": str(e)}
        
        # Create embedding map
        embedding_map = {
            chunk_id: {"vector": vector, "model": model_name} 
            for chunk_id, vector in zip(chunk_ids, embeddings)
        }
        
        # Save embeddings
        save_embeddings(db, embedding_map, model.dimensions)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Embeddings generated in {processing_time:.2f} seconds")
        
        # Update document status
        update_document_status(db, document_id, "embedding", "completed")
        
        # Update document metadata with embedding information
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            if document.metadata:
                document.metadata.update({
                    "embedding": {
                        "model": model_name,
                        "chunks_processed": len(chunks),
                        "processing_time": processing_time
                    }
                })
            else:
                document.metadata = {
                    "embedding": {
                        "model": model_name,
                        "chunks_processed": len(chunks),
                        "processing_time": processing_time
                    }
                }
            db.commit()
        
        return {
            "success": True,
            "document_id": document_id,
            "model": model_name,
            "chunks_processed": len(chunks),
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {str(e)}", exc_info=True)
        update_document_status(db, document_id, "embedding", "failed", str(e))
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def get_chunks(db: Session, document_id: str) -> List[Chunk]:
    """
    Get document chunks from database.
    
    Args:
        db: Database session
        document_id: ID of the document
        
    Returns:
        List of chunks
    """
    return db.query(Chunk).filter(Chunk.document_id == document_id).all()


def save_embeddings(db: Session, embedding_map: Dict[str, Dict[str, Any]], dimensions: int) -> bool:
    """
    Save generated embeddings to database.
    
    Args:
        db: Database session
        embedding_map: Dict mapping chunk IDs to embeddings
        dimensions: Vector dimensions
        
    Returns:
        Success status
    """
    try:
        # Create embedding objects
        embedding_objects = []
        
        for chunk_id, data in embedding_map.items():
            vector = data["vector"]
            model = data["model"]
            
            # Check if embedding already exists
            existing = db.query(Embedding).filter(
                Embedding.chunk_id == chunk_id,
                Embedding.model_id == model
            ).first()
            
            if existing:
                # Update existing embedding
                existing.embedding = vector
                db.flush()
            else:
                # Create new embedding
                embedding = Embedding(
                    id=str(uuid.uuid4()),
                    chunk_id=chunk_id,
                    model_id=model,
                    embedding=vector
                )
                embedding_objects.append(embedding)
        
        # Add to database if any new embeddings
        if embedding_objects:
            db.add_all(embedding_objects)
        
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}", exc_info=True)
        db.rollback()
        return False


def update_document_status(
    db: Session, 
    document_id: str, 
    process_type: str = "embedding", 
    status: str = "completed",
    error_message: Optional[str] = None
) -> bool:
    """
    Update document processing status.
    
    Args:
        db: Database session
        document_id: ID of the document
        process_type: Process type
        status: New status
        error_message: Optional error message
        
    Returns:
        Success status
    """
    try:
        # Get processing status
        processing_status = db.query(ProcessingStatus).filter(
            ProcessingStatus.document_id == document_id,
            ProcessingStatus.process_type == process_type
        ).first()
        
        if processing_status:
            # Update status
            processing_status.status = status
            processing_status.error_message = error_message
        else:
            # Create new status
            processing_status = ProcessingStatus(
                document_id=document_id,
                process_type=process_type,
                status=status,
                error_message=error_message
            )
            db.add(processing_status)
        
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating document status: {str(e)}", exc_info=True)
        db.rollback()
        return False


def compute_embedding_cache_key(text: str, model_name: str) -> str:
    """
    Compute a cache key for an embedding request.
    
    Args:
        text: Text to embed
        model_name: Embedding model name
        
    Returns:
        Cache key
    """
    # Create a hash of the text and model name
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{content_hash}"


def get_cached_embedding(db: Session, cache_key: str) -> Optional[List[float]]:
    """
    Get embedding from cache.
    
    Args:
        db: Database session
        cache_key: Cache key
        
    Returns:
        Cached embedding vector, if found
    """
    try:
        # Query the cache table
        result = db.execute(
            text("SELECT embedding FROM embedding_cache WHERE cache_key = :key"),
            {"key": cache_key}
        ).fetchone()
        
        if result:
            return result[0]  # pgvector column
        return None
    except Exception as e:
        logger.error(f"Error getting cached embedding: {str(e)}", exc_info=True)
        return None


def cache_embedding(db: Session, cache_key: str, embedding: List[float]) -> bool:
    """
    Cache an embedding vector.
    
    Args:
        db: Database session
        cache_key: Cache key
        embedding: Embedding vector
        
    Returns:
        Success status
    """
    try:
        # Check if key exists
        existing = db.execute(
            text("SELECT 1 FROM embedding_cache WHERE cache_key = :key"),
            {"key": cache_key}
        ).fetchone()
        
        if existing:
            # Update existing entry
            db.execute(
                text("UPDATE embedding_cache SET embedding = :embedding WHERE cache_key = :key"),
                {"key": cache_key, "embedding": embedding}
            )
        else:
            # Insert new entry
            db.execute(
                text("INSERT INTO embedding_cache (cache_key, embedding) VALUES (:key, :embedding)"),
                {"key": cache_key, "embedding": embedding}
            )
        
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error caching embedding: {str(e)}", exc_info=True)
        db.rollback()
        return False


def get_embedding(text: str, model_name: str = "text-embedding-ada-002") -> np.ndarray:
    """Get embedding vector for a text using specified model.
    
    Args:
        text: Text to embed
        model_name: Name of the embedding model to use
        
    Returns:
        Embedding vector as numpy array
    """
    try:
        model = get_embedding_model(model_name)
        vector = model.embed(text)
        return np.array(vector)
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        return np.zeros(get_embedding_model(model_name).dimensions)


def generate_embeddings(document_id: str, model_name: str = "text-embedding-ada-002") -> Dict[str, Any]:
    """Generate embeddings for all chunks in a document.
    
    Args:
        document_id: ID of the document to process
        model_name: Name of the embedding model to use
        
    Returns:
        Dictionary with generation results
    """
    try:
        # Get document chunks
        chunks = get_chunks(document_id)
        
        if not chunks:
            return {
                "status": "error",
                "error": "No chunks found",
                "document_id": document_id
            }
        
        # Get embedding model
        model = get_embedding_model(model_name)
        
        # Generate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            vector = model.embed(chunk["text"])
            embeddings.append({
                "chunk_id": chunk["id"],
                "vector": vector,
                "position": chunk["position"]
            })
        
        # Save embeddings
        save_embeddings(document_id, embeddings, model_name)
        
        # Update document status
        update_document_status(document_id, "embeddings_generated")
        
        return {
            "status": "success",
            "document_id": document_id,
            "model": model_name,
            "chunks_processed": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "document_id": document_id
        }

