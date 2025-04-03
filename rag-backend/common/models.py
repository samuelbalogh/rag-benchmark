import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from common.database import Base


class Document(Base):
    """Document model for storing original documents."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    processing_statuses = relationship("ProcessingStatus", back_populates="document", cascade="all, delete-orphan")
    graphs = relationship("GraphMetadata", back_populates="corpus", cascade="all, delete-orphan")


class Chunk(Base):
    """Chunk model for storing document segments."""
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    metadata = Column(JSONB, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")


class ProcessingStatus(Base):
    """Processing status model to track document processing stages."""
    __tablename__ = "processing_status"

    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    process_type = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="processing_statuses")


class Embedding(Base):
    """Embedding model for storing vector embeddings."""
    __tablename__ = "embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(String, nullable=False)
    # The embedding field is handled specially for pgvector
    # SQLAlchemy doesn't directly support the vector type
    # This will be managed in the repository layer
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    chunk = relationship("Chunk", back_populates="embeddings")


class QueryLog(Base):
    """Query log model for storing user queries."""
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    enhanced_query = Column(Text, nullable=True)
    rag_strategy = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    metrics = relationship("ResponseMetric", back_populates="query", cascade="all, delete-orphan")


class ResponseMetric(Base):
    """Response metrics model for storing evaluation metrics."""
    __tablename__ = "response_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False)
    response = Column(Text, nullable=False)
    relevance_score = Column(Float, nullable=True)
    truthfulness_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    estimated_cost = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    query = relationship("QueryLog", back_populates="metrics")


class ApiKey(Base):
    """API key model for authentication."""
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String, nullable=False, unique=True)
    name = Column(String, nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class GraphMetadata(Base):
    """Graph metadata model for tracking knowledge graphs."""
    __tablename__ = "graph_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    corpus_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    embedding_model = Column(String, nullable=False)
    graph_type = Column(String, nullable=False)
    version = Column(Integer, nullable=False)
    node_count = Column(Integer, nullable=False)
    edge_count = Column(Integer, nullable=False)
    parameters = Column(JSONB, nullable=True)
    file_path = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    corpus = relationship("Document", back_populates="graphs") 