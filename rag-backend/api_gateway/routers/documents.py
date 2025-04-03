import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, Query, HTTPException
from sqlalchemy.orm import Session

from common.auth import get_api_key
from common.database import get_db
from common.errors import ResourceNotFoundError
from common.logging import get_logger, with_logging
from common.models import Document, ApiKey, ProcessingStatus
from common.schemas import (
    DocumentCreate,
    DocumentResponse,
    DocumentListResponse,
    ProcessingStatusResponse,
)

# Create router
router = APIRouter()

# Create logger
logger = get_logger(__name__)


@router.post("/documents", response_model=DocumentResponse)
@with_logging(logger)
async def upload_document(
    title: str = Form(...),
    source: str = Form(...),
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Upload a new document for processing.
    """
    # Read file content
    content = await file.read()
    content_text = content.decode("utf-8")
    
    # Create document record
    document = Document(
        id=uuid.uuid4(),
        title=title,
        source=source,
        content=content_text,
        metadata=metadata if metadata else None,
    )
    
    # Add document to database
    db.add(document)
    db.flush()
    
    # Create processing status records
    processing_types = ["chunking", "embedding", "graph"]
    for process_type in processing_types:
        status = ProcessingStatus(
            document_id=document.id,
            process_type=process_type,
            status="pending",
        )
        db.add(status)
    
    # Commit changes
    db.commit()
    
    # TODO: Trigger async processing task
    
    # Return response
    return document


@router.get("/documents", response_model=DocumentListResponse)
@with_logging(logger)
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    List all documents.
    """
    # Calculate offset
    offset = (page - 1) * limit
    
    # Query documents
    total = db.query(Document).count()
    documents = db.query(Document).order_by(Document.created_at.desc()).offset(offset).limit(limit).all()
    
    # Return response
    return DocumentListResponse(
        items=documents,
        total=total,
        page=page,
        page_size=limit,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
@with_logging(logger)
async def get_document(
    document_id: uuid.UUID,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Get document details.
    """
    # Query document
    document = db.query(Document).filter(Document.id == document_id).first()
    
    # Check if document exists
    if not document:
        raise ResourceNotFoundError(
            message=f"Document with ID {document_id} not found",
            resource_type="document",
            resource_id=str(document_id),
        )
    
    # Return response
    return document


@router.get("/documents/{document_id}/status", response_model=List[ProcessingStatusResponse])
@with_logging(logger)
async def get_document_status(
    document_id: uuid.UUID,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Get document processing status.
    """
    # Check if document exists
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ResourceNotFoundError(
            message=f"Document with ID {document_id} not found",
            resource_type="document",
            resource_id=str(document_id),
        )
    
    # Query processing status
    statuses = db.query(ProcessingStatus).filter(ProcessingStatus.document_id == document_id).all()
    
    # Return response
    return statuses


@router.delete("/documents/{document_id}")
@with_logging(logger)
async def delete_document(
    document_id: uuid.UUID,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Delete a document and all associated data.
    """
    # Check if document exists
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise ResourceNotFoundError(
            message=f"Document with ID {document_id} not found",
            resource_type="document",
            resource_id=str(document_id),
        )
    
    # Delete document
    db.delete(document)
    db.commit()
    
    # Return response
    return {"message": "Document deleted successfully"} 