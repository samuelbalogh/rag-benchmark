from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


# Document schemas
class DocumentBase(BaseModel):
    title: str
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int


# Chunk schemas
class ChunkBase(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    chunk_index: int


class ChunkCreate(ChunkBase):
    document_id: UUID


class ChunkResponse(ChunkBase):
    id: UUID
    document_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# ProcessingStatus schemas
class ProcessingStatusBase(BaseModel):
    process_type: str
    status: str
    error_message: Optional[str] = None


class ProcessingStatusCreate(ProcessingStatusBase):
    document_id: UUID


class ProcessingStatusResponse(ProcessingStatusBase):
    document_id: UUID
    updated_at: datetime

    class Config:
        from_attributes = True


# Embedding schemas
class EmbeddingBase(BaseModel):
    model_id: str


class EmbeddingCreate(EmbeddingBase):
    chunk_id: UUID
    embedding: List[float]  # Vector will be handled specially


class EmbeddingResponse(EmbeddingBase):
    id: UUID
    chunk_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Query schemas
class QueryBase(BaseModel):
    query: str
    rag_strategy: str = "vector"  # Default to vector RAG


class QueryCreate(QueryBase):
    pass


class QueryResponse(QueryBase):
    id: UUID
    enhanced_query: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Response metrics schemas
class ResponseMetricBase(BaseModel):
    response: str
    relevance_score: Optional[float] = None
    truthfulness_score: Optional[float] = None
    completeness_score: Optional[float] = None
    latency_ms: int
    token_count: int
    estimated_cost: float


class ResponseMetricCreate(ResponseMetricBase):
    query_id: UUID


class ResponseMetricResponse(ResponseMetricBase):
    id: UUID
    query_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# API key schemas
class ApiKeyCreate(BaseModel):
    name: str


class ApiKeyResponse(BaseModel):
    id: UUID
    key: str
    name: str
    enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True


# Graph metadata schemas
class GraphMetadataBase(BaseModel):
    embedding_model: str
    graph_type: str
    version: int
    node_count: int
    edge_count: int
    parameters: Optional[Dict[str, Any]] = None
    file_path: str


class GraphMetadataCreate(GraphMetadataBase):
    corpus_id: UUID


class GraphMetadataResponse(GraphMetadataBase):
    id: UUID
    corpus_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True


# Query enhancement schemas
class QueryEnhancementRequest(BaseModel):
    query: str
    enhancement_strategies: Optional[List[str]] = Field(default_factory=list)


class QueryEnhancementResponse(BaseModel):
    original_query: str
    enhanced_query: str
    applied_strategies: List[str]


# RAG query and response schemas
class RAGQueryRequest(BaseModel):
    query: str
    rag_strategy: str = "vector"
    embedding_model: str = "text-embedding-3-small"
    use_enhancement: bool = True
    enhancement_strategies: Optional[List[str]] = None
    max_context_chunks: int = 5


class ContextChunk(BaseModel):
    content: str
    document_title: str
    source: str
    metadata: Optional[Dict[str, Any]] = None
    relevance_score: Optional[float] = None


class RAGQueryResponse(BaseModel):
    query: str
    enhanced_query: Optional[str] = None
    rag_strategy: str
    response: str
    context_chunks: List[ContextChunk]
    metrics: ResponseMetricResponse
    
    
# Benchmark schemas
class BenchmarkQuestion(BaseModel):
    question: str
    expected_answer: Optional[str] = None
    tags: Optional[List[str]] = None


class BenchmarkConfig(BaseModel):
    questions: List[BenchmarkQuestion]
    rag_strategies: List[str]
    embedding_models: List[str]
    use_enhancement: bool = True


class BenchmarkResult(BaseModel):
    strategy: str
    embedding_model: str
    use_enhancement: bool
    avg_relevance: Optional[float] = None
    avg_truthfulness: Optional[float] = None
    avg_completeness: Optional[float] = None
    avg_latency_ms: int
    avg_token_count: int
    avg_cost: float


class BenchmarkResponse(BaseModel):
    results: List[BenchmarkResult]
    questions_count: int
    completed_at: datetime 