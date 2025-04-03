import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from common.auth import get_api_key
from common.database import get_db
from common.errors import ResourceNotFoundError, ValidationError
from common.logging import get_logger, with_logging
from common.models import ApiKey, QueryLog
from common.schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    BenchmarkConfig,
    BenchmarkResponse,
    QueryResponse,
)

# Create router
router = APIRouter()

# Create logger
logger = get_logger(__name__)


@router.post("/query", response_model=RAGQueryResponse)
@with_logging(logger)
async def process_query(
    query_request: RAGQueryRequest,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Process a query using specified RAG strategy.
    """
    # Log query
    logger.info(
        f"Processing query with strategy: {query_request.rag_strategy}",
        extra={
            "query": query_request.query,
            "rag_strategy": query_request.rag_strategy,
            "embedding_model": query_request.embedding_model,
            "use_enhancement": query_request.use_enhancement,
        },
    )
    
    # Validate RAG strategy
    valid_strategies = ["vector", "graph", "hybrid"]
    if query_request.rag_strategy not in valid_strategies:
        raise ValidationError(
            message=f"Invalid RAG strategy: {query_request.rag_strategy}",
            field_errors={"rag_strategy": [f"Must be one of: {', '.join(valid_strategies)}"]},
        )
    
    # Validate embedding model
    valid_models = ["ada", "text-embedding-3-small", "text-embedding-3-large", "voyage"]
    if query_request.embedding_model not in valid_models:
        raise ValidationError(
            message=f"Invalid embedding model: {query_request.embedding_model}",
            field_errors={"embedding_model": [f"Must be one of: {', '.join(valid_models)}"]},
        )
    
    # Create query log
    query_log = QueryLog(
        id=uuid.uuid4(),
        query=query_request.query,
        rag_strategy=query_request.rag_strategy,
    )
    db.add(query_log)
    db.commit()
    
    # TODO: This is a placeholder - in a real implementation, we would call:
    # 1. Query enhancement service if use_enhancement is True
    # 2. RAG orchestrator with appropriate strategy
    # 3. Evaluation service for metrics
    
    # For now, return mock response
    return RAGQueryResponse(
        query=query_request.query,
        enhanced_query="What is the One Ring in Lord of the Rings?" if query_request.use_enhancement else None,
        rag_strategy=query_request.rag_strategy,
        response="The One Ring is a central element in J.R.R. Tolkien's 'The Lord of the Rings'. It was created by the Dark Lord Sauron during the Second Age to gain dominion over the other Rings of Power and through them, Middle-earth. The Ring granted invisibility to its bearer but also corrupted them.",
        context_chunks=[
            {
                "content": "The One Ring was created by the Dark Lord Sauron during the Second Age in order to gain dominion over the free peoples of Middle-earth. In disguise as Annatar, or 'Lord of Gifts', he aided the Elven smiths of Eregion and their leader Celebrimbor in the making of the Rings of Power.",
                "document_title": "The Lord of the Rings",
                "source": "Book",
                "metadata": {"page": 1, "chapter": "Prologue"},
                "relevance_score": 0.92,
            },
            {
                "content": "The One Ring was forged in the fires of Mount Doom, and it could only be destroyed there. It had a will of its own and would attempt to return to Sauron. It would also try to corrupt its bearer, regardless of their initial intent.",
                "document_title": "The Lord of the Rings",
                "source": "Book",
                "metadata": {"page": 2, "chapter": "Prologue"},
                "relevance_score": 0.88,
            },
        ],
        metrics={
            "id": uuid.uuid4(),
            "query_id": query_log.id,
            "response": "The One Ring is a central element in J.R.R. Tolkien's 'The Lord of the Rings'. It was created by the Dark Lord Sauron during the Second Age to gain dominion over the other Rings of Power and through them, Middle-earth. The Ring granted invisibility to its bearer but also corrupted them.",
            "relevance_score": 0.85,
            "truthfulness_score": 0.92,
            "completeness_score": 0.78,
            "latency_ms": 350,
            "token_count": 73,
            "estimated_cost": 0.0005,
            "created_at": datetime.utcnow(),
        },
    )


@router.get("/query/history", response_model=List[QueryResponse])
@with_logging(logger)
async def get_query_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Get query history.
    """
    # Query recent queries
    queries = db.query(QueryLog).order_by(QueryLog.created_at.desc()).offset(offset).limit(limit).all()
    
    # Return response
    return queries


@router.post("/query/benchmark", response_model=BenchmarkResponse)
@with_logging(logger)
async def run_benchmark(
    benchmark_config: BenchmarkConfig,
    background_tasks: BackgroundTasks,
    api_key: ApiKey = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    """
    Run benchmark with predefined questions.
    """
    # Validate RAG strategies
    valid_strategies = ["vector", "graph", "hybrid"]
    for strategy in benchmark_config.rag_strategies:
        if strategy not in valid_strategies:
            raise ValidationError(
                message=f"Invalid RAG strategy: {strategy}",
                field_errors={"rag_strategies": [f"Must contain only: {', '.join(valid_strategies)}"]},
            )
    
    # Validate embedding models
    valid_models = ["ada", "text-embedding-3-small", "text-embedding-3-large", "voyage"]
    for model in benchmark_config.embedding_models:
        if model not in valid_models:
            raise ValidationError(
                message=f"Invalid embedding model: {model}",
                field_errors={"embedding_models": [f"Must contain only: {', '.join(valid_models)}"]},
            )
    
    # Check if at least one question is provided
    if not benchmark_config.questions:
        raise ValidationError(
            message="No questions provided",
            field_errors={"questions": ["At least one question is required"]},
        )
    
    # TODO: Run benchmark in background task
    
    # For now, return mock response
    results = []
    for strategy in benchmark_config.rag_strategies:
        for model in benchmark_config.embedding_models:
            results.append({
                "strategy": strategy,
                "embedding_model": model,
                "use_enhancement": benchmark_config.use_enhancement,
                "avg_relevance": 0.82,
                "avg_truthfulness": 0.90,
                "avg_completeness": 0.75,
                "avg_latency_ms": 320,
                "avg_token_count": 68,
                "avg_cost": 0.0004,
            })
    
    return BenchmarkResponse(
        results=results,
        questions_count=len(benchmark_config.questions),
        completed_at=datetime.utcnow(),
    ) 