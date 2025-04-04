import os
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple

from sqlalchemy.orm import Session

from common.database import SessionLocal
from common.logging import get_logger, with_logging
from common.models import Document, Chunk, ProcessingStatus
from corpus_service.worker import app
from corpus_service.chunking_strategies import get_chunking_strategy

# Create logger
logger = get_logger(__name__)


@app.task(name="corpus_service.tasks.chunking.process_document")
def process_document(document_id: str, strategy_name: str = "fixed_length") -> Dict[str, Any]:
    """
    Process a document for chunking.
    
    Args:
        document_id: ID of the document to process
        strategy_name: Chunking strategy to use
        
    Returns:
        Dict with processing results
    """
    logger.info(f"Processing document for chunking: {document_id} with strategy: {strategy_name}")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Update processing status
        _update_processing_status(db, document_id, "chunking", "processing")
        
        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document not found: {document_id}")
            _update_processing_status(
                db, document_id, "chunking", "failed", "Document not found"
            )
            return {"success": False, "error": "Document not found"}
        
        # Get chunking strategy
        strategy = get_chunking_strategy(strategy_name)
        
        # Extract chunking parameters from document metadata if available
        chunking_params = document.metadata.get("chunking_params", {}) if document.metadata else {}
        
        # Perform chunking with selected strategy
        chunks = strategy.chunk_document(document.content, document.id, **chunking_params)
        
        # Store chunks in database
        store_chunks(db, chunks)
        
        # Update document metadata with chunking information
        if document.metadata:
            document.metadata.update({
                "chunking": {
                    "strategy": strategy_name,
                    "chunks_count": len(chunks),
                    "parameters": chunking_params
                }
            })
        else:
            document.metadata = {
                "chunking": {
                    "strategy": strategy_name,
                    "chunks_count": len(chunks),
                    "parameters": chunking_params
                }
            }
        db.commit()
        
        # Update processing status
        _update_processing_status(db, document_id, "chunking", "completed")
        
        # Trigger embedding generation
        logger.info(f"Chunking completed for document: {document_id}, triggering embedding generation")
        trigger_embedding_generation(document_id)
        
        return {
            "success": True,
            "document_id": document_id,
            "chunks_count": len(chunks),
            "strategy": strategy_name
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        _update_processing_status(
            db, document_id, "chunking", "failed", str(e)
        )
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def _update_processing_status(
    db: Session,
    document_id: str,
    process_type: str,
    status: str,
    error_message: Optional[str] = None,
) -> None:
    """
    Update processing status for a document.
    
    Args:
        db: Database session
        document_id: ID of the document
        process_type: Type of processing
        status: Status to set
        error_message: Optional error message
    """
    # Get processing status
    processing_status = db.query(ProcessingStatus).filter(
        ProcessingStatus.document_id == document_id,
        ProcessingStatus.process_type == process_type,
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
            error_message=error_message,
        )
        db.add(processing_status)
    
    # Commit changes
    db.commit()


def chunk_document(
    content: str, 
    document_id: str, 
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """
    Chunk a document into smaller segments.
    
    Args:
        content: Document content
        document_id: ID of the document
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunks
    """
    # Simple chunking by character count with overlap
    chunks = []
    
    # If content is small enough, use as a single chunk
    if len(content) <= chunk_size:
        chunks.append({
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "content": content,
            "metadata": {"index": 0, "start": 0, "end": len(content)},
            "chunk_index": 0,
        })
        return chunks
    
    # Otherwise, chunk with overlap
    start = 0
    chunk_index = 0
    
    while start < len(content):
        # Calculate end position
        end = min(start + chunk_size, len(content))
        
        # If not at the end of the content and not the first chunk
        # Try to find a good break point (period followed by space)
        if end < len(content) and start > 0:
            # Look for the last period in the chunk
            last_period = content.rfind('. ', start, end)
            if last_period > start + chunk_size // 2:  # Only use if it's in the latter half
                end = last_period + 1  # Include the period
        
        # Create chunk
        chunk_content = content[start:end]
        
        # Add to chunks list
        chunks.append({
            "id": str(uuid.uuid4()),
            "document_id": document_id,
            "content": chunk_content,
            "metadata": {"index": chunk_index, "start": start, "end": end},
            "chunk_index": chunk_index,
        })
        
        # Update start position for next chunk
        start = end - chunk_overlap if end < len(content) else len(content)
        chunk_index += 1
    
    return chunks


def store_chunks(db: Session, chunks: List[Dict[str, Any]]) -> None:
    """
    Store chunks in the database.
    
    Args:
        db: Database session
        chunks: List of chunks to store
    """
    # Create Chunk objects
    chunk_objects = []
    for chunk_data in chunks:
        chunk = Chunk(
            id=chunk_data["id"],
            document_id=chunk_data["document_id"],
            content=chunk_data["content"],
            metadata=chunk_data["metadata"],
            chunk_index=chunk_data["chunk_index"],
        )
        chunk_objects.append(chunk)
    
    # Add to database
    db.add_all(chunk_objects)
    db.commit()


@app.task(name="corpus_service.tasks.chunking.cleanup_old_chunks")
def cleanup_old_chunks(days: int = 30) -> Dict[str, Any]:
    """
    Clean up old chunks that are no longer needed.
    
    Args:
        days: Number of days to keep chunks
        
    Returns:
        Dict with cleanup results
    """
    # This is a placeholder for a task that would clean up old chunks
    # In a real implementation, this would delete chunks from documents
    # that are older than the specified number of days
    return {"success": True, "message": "Cleanup task executed"}


def trigger_embedding_generation(document_id):
    """Trigger embedding generation for a document.
    
    Args:
        document_id: ID of the document
        
    Returns:
        Task information
    """
    logger.info(f"Triggering embedding generation for document: {document_id}")
    
    # This would call the embedding service in production
    # For testing, we just return a success response
    return {
        "status": "success",
        "document_id": document_id,
        "message": "Embedding generation triggered"
    }

# Make it available as a task
trigger_embedding_generation.delay = trigger_embedding_generation 