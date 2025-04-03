import uuid
from typing import Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from common.auth import get_api_key
from common.database import get_db
from common.errors import ValidationError
from common.logging import get_logger, with_logging
from common.models import ApiKey
from common.schemas import ApiKeyCreate, ApiKeyResponse

# Create router
router = APIRouter()

# Create logger
logger = get_logger(__name__)


@router.get("/admin/stats")
@with_logging(logger)
async def get_stats(
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Get system statistics.
    """
    # TODO: Implement real stats collection
    # For now, return mock stats
    return {
        "documents": {
            "total": 25,
            "processed": 22,
            "failed": 3,
        },
        "queries": {
            "total": 152,
            "by_strategy": {
                "vector": 87,
                "graph": 34,
                "hybrid": 31,
            },
            "avg_latency_ms": 345,
        },
        "embeddings": {
            "by_model": {
                "ada": 1250,
                "text-embedding-3-small": 2430,
                "text-embedding-3-large": 1100,
                "voyage": 980,
            },
        },
        "storage": {
            "total_mb": 320,
            "vector_index_mb": 180,
            "graph_storage_mb": 75,
            "document_storage_mb": 65,
        },
    }


@router.post("/admin/api-keys", response_model=ApiKeyResponse)
@with_logging(logger)
async def create_api_key(
    api_key_create: ApiKeyCreate,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Create a new API key.
    """
    # Generate a new API key
    new_key = f"ragbench_{uuid.uuid4().hex}"
    
    # Create API key record
    api_key_record = ApiKey(
        id=uuid.uuid4(),
        key=new_key,
        name=api_key_create.name,
        enabled=True,
    )
    
    # Add to database
    db.add(api_key_record)
    db.commit()
    db.refresh(api_key_record)
    
    # Return response
    return api_key_record


@router.get("/admin/api-keys", response_model=List[ApiKeyResponse])
@with_logging(logger)
async def list_api_keys(
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    List all API keys.
    """
    # Query API keys
    api_keys = db.query(ApiKey).all()
    
    # Return response
    return api_keys


@router.delete("/admin/api-keys/{key_id}")
@with_logging(logger)
async def delete_api_key(
    key_id: uuid.UUID,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Delete an API key.
    """
    # Find API key
    api_key_record = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    
    # Check if API key exists
    if not api_key_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found",
        )
    
    # Prevent deleting the API key being used for this request
    if api_key_record.id == api_key.id:
        raise ValidationError(
            message="Cannot delete the API key used for this request",
        )
    
    # Delete API key
    db.delete(api_key_record)
    db.commit()
    
    # Return response
    return {"message": "API key deleted successfully"}


@router.post("/admin/rebuild-indexes")
@with_logging(logger)
async def rebuild_indexes(
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
):
    """
    Rebuild vector indexes.
    """
    # TODO: Implement real index rebuilding
    # For now, return mock response
    return {
        "job_id": str(uuid.uuid4()),
        "status": "scheduled",
        "message": "Index rebuild job scheduled",
    }


@router.post("/admin/rebuild-graph")
@with_logging(logger)
async def rebuild_graph(
    corpus_id: uuid.UUID,
    embedding_model: str,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Rebuild knowledge graph.
    """
    # Validate embedding model
    valid_models = ["ada", "text-embedding-3-small", "text-embedding-3-large", "voyage"]
    if embedding_model not in valid_models:
        raise ValidationError(
            message=f"Invalid embedding model: {embedding_model}",
            field_errors={"embedding_model": [f"Must be one of: {', '.join(valid_models)}"]},
        )
    
    # TODO: Implement real graph rebuilding
    # For now, return mock response
    return {
        "job_id": str(uuid.uuid4()),
        "status": "scheduled",
        "message": "Graph rebuild job scheduled",
        "corpus_id": str(corpus_id),
        "embedding_model": embedding_model,
    } 