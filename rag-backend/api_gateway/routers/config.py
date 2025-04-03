from typing import Dict, List, Any

from fastapi import APIRouter, Depends

from common.auth import get_api_key
from common.logging import get_logger, with_logging
from common.models import ApiKey

# Create router
router = APIRouter()

# Create logger
logger = get_logger(__name__)


@router.get("/config/embedding-models")
@with_logging(logger)
async def get_embedding_models(
    api_key: ApiKey = Depends(get_api_key),
):
    """
    List available embedding models.
    """
    # Return available embedding models with details
    return {
        "models": [
            {
                "id": "ada",
                "name": "OpenAI ADA (legacy)",
                "dimensions": 1536,
                "description": "Legacy OpenAI ada embedding model",
                "performance_profile": {
                    "quality": "moderate",
                    "speed": "fast",
                    "cost": "low",
                },
            },
            {
                "id": "text-embedding-3-small",
                "name": "OpenAI text-embedding-3-small",
                "dimensions": 1536,
                "description": "Modern OpenAI small embedding model with good balance of performance/cost",
                "performance_profile": {
                    "quality": "good",
                    "speed": "fast",
                    "cost": "moderate",
                },
            },
            {
                "id": "text-embedding-3-large",
                "name": "OpenAI text-embedding-3-large",
                "dimensions": 3072,
                "description": "High-quality OpenAI large embedding model for best performance",
                "performance_profile": {
                    "quality": "excellent",
                    "speed": "moderate",
                    "cost": "high",
                },
            },
            {
                "id": "voyage",
                "name": "Voyage AI Embeddings",
                "dimensions": 1024,
                "description": "High-quality embeddings from Voyage AI",
                "performance_profile": {
                    "quality": "excellent",
                    "speed": "moderate",
                    "cost": "high",
                },
            },
        ]
    }


@router.get("/config/rag-strategies")
@with_logging(logger)
async def get_rag_strategies(
    api_key: ApiKey = Depends(get_api_key),
):
    """
    List available RAG strategies.
    """
    # Return available RAG strategies with details
    return {
        "strategies": [
            {
                "id": "vector",
                "name": "Vector Search RAG",
                "description": "Basic RAG using vector search for similarity matching",
                "parameters": {
                    "max_chunks": {
                        "description": "Maximum number of chunks to retrieve",
                        "type": "integer",
                        "default": 5,
                        "min": 1,
                        "max": 20,
                    },
                    "similarity_threshold": {
                        "description": "Minimum similarity score for chunks",
                        "type": "float",
                        "default": 0.7,
                        "min": 0.1,
                        "max": 0.99,
                    },
                },
                "use_cases": ["Simple factual queries", "When speed is critical"],
            },
            {
                "id": "graph",
                "name": "Knowledge Graph RAG",
                "description": "Advanced RAG using knowledge graph for entity-centric retrieval",
                "parameters": {
                    "max_hops": {
                        "description": "Maximum traversal hops in the graph",
                        "type": "integer",
                        "default": 2,
                        "min": 1,
                        "max": 5,
                    },
                    "centrality_weight": {
                        "description": "Weight given to node centrality",
                        "type": "float",
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                    },
                },
                "use_cases": ["Multi-hop questions", "Entity-relationship queries"],
            },
            {
                "id": "hybrid",
                "name": "Hybrid RAG",
                "description": "Combined approach using both vector search and knowledge graph",
                "parameters": {
                    "vector_weight": {
                        "description": "Weight given to vector search results",
                        "type": "float",
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                    },
                    "max_chunks": {
                        "description": "Maximum number of chunks to retrieve",
                        "type": "integer",
                        "default": 8,
                        "min": 1,
                        "max": 30,
                    },
                },
                "use_cases": ["Complex questions", "Best overall performance"],
            },
        ]
    }


@router.get("/config/query-enhancement-strategies")
@with_logging(logger)
async def get_query_enhancement_strategies(
    api_key: ApiKey = Depends(get_api_key),
):
    """
    List available query enhancement strategies.
    """
    # Return available query enhancement strategies with details
    return {
        "strategies": [
            {
                "id": "synonym_expansion",
                "name": "Synonym Expansion",
                "description": "Expand query with synonyms of key terms",
                "parameters": {
                    "max_synonyms": {
                        "description": "Maximum synonyms per term",
                        "type": "integer",
                        "default": 3,
                        "min": 1,
                        "max": 10,
                    },
                },
            },
            {
                "id": "hypothetical_document",
                "name": "Hypothetical Document",
                "description": "Generate a hypothetical perfect document for the query",
                "parameters": {
                    "token_limit": {
                        "description": "Maximum tokens for hypothetical document",
                        "type": "integer",
                        "default": 100,
                        "min": 50,
                        "max": 500,
                    },
                },
            },
            {
                "id": "query_decomposition",
                "name": "Query Decomposition",
                "description": "Break complex queries into simpler subqueries",
                "parameters": {
                    "max_subqueries": {
                        "description": "Maximum number of subqueries",
                        "type": "integer",
                        "default": 3,
                        "min": 2,
                        "max": 5,
                    },
                },
            },
            {
                "id": "llm_rewrite",
                "name": "LLM Query Rewrite",
                "description": "Use LLM to rephrase the query for better retrieval",
                "parameters": {
                    "model": {
                        "description": "LLM model to use",
                        "type": "string",
                        "default": "gpt-3.5-turbo",
                        "options": ["gpt-3.5-turbo", "gpt-4o"],
                    },
                },
            },
        ]
    } 