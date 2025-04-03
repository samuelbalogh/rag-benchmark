import os
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from common.database import SessionLocal
from common.logging import get_logger, with_logging
from common.models import Document, Chunk, ProcessingStatus
from corpus_service.worker import app

# Create logger
logger = get_logger(__name__)


@app.task(name="corpus_service.tasks.processing.process_document_pipeline")
def process_document_pipeline(document_id: str) -> Dict[str, Any]:
    """
    Process document through the entire pipeline.
    
    Args:
        document_id: ID of the document to process
        
    Returns:
        Dict with processing results
    """
    logger.info(f"Starting document processing pipeline for: {document_id}")
    
    # Kick off chunking task
    from corpus_service.tasks.chunking import process_document
    chunking_result = process_document(document_id)
    
    if not chunking_result.get("success", False):
        logger.error(f"Chunking failed for document: {document_id}")
        return {
            "success": False,
            "error": chunking_result.get("error", "Chunking task failed"),
            "document_id": document_id,
        }
    
    # Here we would kick off the embedding task
    # This would typically be handled by a separate service
    # For now, we'll just log the completion
    logger.info(f"Chunking completed for document: {document_id}")
    
    return {
        "success": True,
        "document_id": document_id,
        "message": "Document processing pipeline started",
    }


@app.task(name="corpus_service.tasks.processing.retry_failed_tasks")
def retry_failed_tasks(process_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Retry failed processing tasks.
    
    Args:
        process_type: Optional process type to filter by
        
    Returns:
        Dict with retry results
    """
    logger.info("Retrying failed processing tasks")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Query for failed tasks
        query = db.query(ProcessingStatus).filter(ProcessingStatus.status == "failed")
        
        if process_type:
            query = query.filter(ProcessingStatus.process_type == process_type)
        
        failed_tasks = query.all()
        
        # Count of retried tasks
        retried = 0
        
        # Retry each failed task
        for task in failed_tasks:
            logger.info(f"Retrying {task.process_type} for document: {task.document_id}")
            
            # Update status to pending
            task.status = "pending"
            task.error_message = None
            task.updated_at = datetime.utcnow()
            db.commit()
            
            # Kick off appropriate task based on process type
            if task.process_type == "chunking":
                from corpus_service.tasks.chunking import process_document
                process_document.delay(str(task.document_id))
                retried += 1
            
            # Add other process types as they are implemented
            
        return {
            "success": True,
            "message": f"Retried {retried} failed tasks",
            "retried_count": retried,
            "process_type": process_type or "all",
        }
    except Exception as e:
        logger.error(f"Error retrying failed tasks: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "process_type": process_type or "all",
        }
    finally:
        db.close()


@app.task(name="corpus_service.tasks.processing.get_processing_stats")
def get_processing_stats() -> Dict[str, Any]:
    """
    Get document processing statistics.
    
    Returns:
        Dict with processing stats
    """
    logger.info("Getting document processing statistics")
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Count documents by status for each process type
        stats = {}
        process_types = ["chunking", "embedding", "graph"]
        
        for process_type in process_types:
            # Query counts for each status
            pending = db.query(func.count(ProcessingStatus.document_id)).filter(
                ProcessingStatus.process_type == process_type,
                ProcessingStatus.status == "pending",
            ).scalar() or 0
            
            processing = db.query(func.count(ProcessingStatus.document_id)).filter(
                ProcessingStatus.process_type == process_type,
                ProcessingStatus.status == "processing",
            ).scalar() or 0
            
            completed = db.query(func.count(ProcessingStatus.document_id)).filter(
                ProcessingStatus.process_type == process_type,
                ProcessingStatus.status == "completed",
            ).scalar() or 0
            
            failed = db.query(func.count(ProcessingStatus.document_id)).filter(
                ProcessingStatus.process_type == process_type,
                ProcessingStatus.status == "failed",
            ).scalar() or 0
            
            stats[process_type] = {
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "total": pending + processing + completed + failed,
            }
        
        # Count total documents
        document_count = db.query(func.count(Document.id)).scalar() or 0
        
        # Count total chunks
        chunk_count = db.query(func.count(Chunk.id)).scalar() or 0
        
        return {
            "success": True,
            "process_stats": stats,
            "document_count": document_count,
            "chunk_count": chunk_count,
        }
    except Exception as e:
        logger.error(f"Error getting processing stats: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        db.close()


@app.task(name="corpus_service.tasks.processing.cleanup_old_tasks")
def cleanup_old_tasks(days: int = 30) -> Dict[str, Any]:
    """
    Cleanup old tasks from status tables.
    
    Args:
        days: Number of days to keep records
        
    Returns:
        Dict with cleanup results
    """
    logger.info(f"Cleaning up processing tasks older than {days} days")
    
    # Calculate cutoff date
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Query for documents older than cutoff
        old_docs = db.query(Document.id).filter(Document.created_at < cutoff_date).all()
        old_doc_ids = [str(doc[0]) for doc in old_docs]
        
        # Log number of old documents
        logger.info(f"Found {len(old_doc_ids)} old documents for cleanup")
        
        # If no old documents, return
        if not old_doc_ids:
            return {
                "success": True,
                "message": "No old documents found for cleanup",
                "deleted_count": 0,
            }
        
        # Delete old processing status entries
        deleted_count = 0
        for doc_id in old_doc_ids:
            # Skip documents that are still in process
            in_process = db.query(ProcessingStatus).filter(
                ProcessingStatus.document_id == doc_id,
                ProcessingStatus.status.in_(["pending", "processing"]),
            ).first()
            
            if in_process:
                logger.info(f"Skipping document still in process: {doc_id}")
                continue
            
            # Delete completed or failed processing entries
            result = db.query(ProcessingStatus).filter(
                ProcessingStatus.document_id == doc_id,
                ProcessingStatus.status.in_(["completed", "failed"]),
            ).delete()
            
            deleted_count += result
        
        # Commit changes
        db.commit()
        
        logger.info(f"Deleted {deleted_count} processing status entries")
        
        return {
            "success": True,
            "message": f"Deleted {deleted_count} processing status entries",
            "deleted_count": deleted_count,
        }
    except Exception as e:
        logger.error(f"Error cleaning up old tasks: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        db.close()


def update_document_status(document_id, status, error_message=None):
    """Update document processing status.
    
    Args:
        document_id: ID of the document
        status: New status
        error_message: Optional error message
        
    Returns:
        Success status
    """
    logger.info(f"Updating document {document_id} status to {status}")
    
    # This would update the database in production
    # For testing, we just return a success response
    return True 