"""Query API routes for the RAG benchmark platform."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from pydantic import BaseModel, Field

from orchestration_service.service import (
    process_query_with_strategy,
    compare_strategies,
    process_complex_query
)
from evaluation_service.service import calculate_metrics
from common.logging import get_logger

# Initialize router
router = APIRouter(prefix="/query", tags=["query"])

# Initialize logger
logger = get_logger(__name__)


class QueryRequest(BaseModel):
    """Request model for query processing."""
    
    query: str = Field(..., description="The query to process")
    strategy_key: str = Field("vector", description="The strategy key to use for processing")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs to search")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Optional parameters for the strategy")
    return_documents: bool = Field(False, description="Whether to include documents in the response")


class ComparisonRequest(BaseModel):
    """Request model for strategy comparison."""
    
    query: str = Field(..., description="The query to process")
    strategy_keys: List[str] = Field(["vector", "knowledge_graph", "hybrid"], description="The strategies to compare")
    document_ids: Optional[List[str]] = Field(None, description="Optional list of document IDs to search")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Optional parameters for the strategies")
    return_documents: bool = Field(False, description="Whether to include documents in the response")


class QueryResponse(BaseModel):
    """Response model for query processing."""
    
    query: str = Field(..., description="The original query")
    enhanced_query: Any = Field(..., description="The enhanced query")
    response: str = Field(..., description="The generated response")
    strategy_key: str = Field(..., description="The strategy used for processing")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Evaluation metrics")
    documents: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved documents")
    document_count: Optional[int] = Field(None, description="Number of retrieved documents")
    processing_time: float = Field(..., description="Processing time in seconds")


@router.post("/process", response_model=QueryResponse, summary="Process a query with a specific strategy")
async def process_query(request: QueryRequest):
    """Process a query using a specific RAG strategy.
    
    This endpoint processes a query using the specified RAG strategy and returns the response,
    along with optional evaluation metrics and retrieved documents.
    
    Args:
        request: The query processing request
        
    Returns:
        The query processing response
    """
    logger.info(f"Processing query with strategy {request.strategy_key}: {request.query}")
    
    try:
        result = process_query_with_strategy(
            query=request.query,
            strategy_key=request.strategy_key,
            document_ids=request.document_ids,
            override_params=request.parameters,
            return_documents=request.return_documents
        )
        
        # Format the response
        response = QueryResponse(
            query=request.query,
            enhanced_query=result.get("enhanced_query", request.query),
            response=result["response"],
            strategy_key=result["strategy_key"],
            metrics=result.get("metrics"),
            processing_time=result["metadata"]["processing_time"]
        )
        
        # Include documents or document count based on request
        if request.return_documents and "documents" in result:
            response.documents = result["documents"]
        else:
            response.document_count = result.get("document_count", 0)
        
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/compare", summary="Compare multiple strategies for a query")
async def compare_query_strategies(request: ComparisonRequest):
    """Compare multiple RAG strategies for a query.
    
    This endpoint processes a query using multiple RAG strategies and returns a comparison
    of the results, including evaluation metrics.
    
    Args:
        request: The strategy comparison request
        
    Returns:
        The strategy comparison results
    """
    logger.info(f"Comparing strategies for query: {request.query}")
    
    try:
        result = compare_strategies(
            query=request.query,
            strategy_keys=request.strategy_keys,
            document_ids=request.document_ids,
            **(request.parameters or {}),
            return_documents=request.return_documents
        )
        
        return result
    except Exception as e:
        logger.error(f"Error comparing strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing strategies: {str(e)}")


@router.post("/adaptive", summary="Process a query using adaptive strategy selection")
async def process_adaptive_query(query: str, return_documents: bool = False):
    """Process a query using adaptive strategy selection.
    
    This endpoint analyzes the query and selects the most appropriate RAG strategy
    based on the query characteristics.
    
    Args:
        query: The query to process
        return_documents: Whether to include documents in the response
        
    Returns:
        The query processing results
    """
    logger.info(f"Processing adaptive query: {query}")
    
    try:
        result = process_complex_query(
            query=query,
            return_documents=return_documents
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing adaptive query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing adaptive query: {str(e)}") 