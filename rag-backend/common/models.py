"""SQLAlchemy models for the RAG benchmark platform."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from enum import Enum

from common.database import Base


class Document(Base):
    """Document model for source documents uploaded by users."""
    
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    file_path = Column(String)
    file_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, index=True)
    status = Column(String, default="uploaded")
    document_meta = Column(JSON, nullable=True)  # Renamed from metadata to avoid conflicts
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document")


class Chunk(Base):
    """Chunk model for segments of documents used for retrieval."""
    
    __tablename__ = "chunks"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"))
    content = Column(Text)
    position = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk")
    entities = relationship("Entity", back_populates="chunk")


class Embedding(Base):
    """Embedding model for vector representations of chunks."""
    
    __tablename__ = "embeddings"
    
    id = Column(String, primary_key=True, index=True)
    chunk_id = Column(String, ForeignKey("chunks.id"))
    model_name = Column(String, index=True)
    vector = Column(Text)  # Stored as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunk = relationship("Chunk", back_populates="embeddings")


class Entity(Base):
    """Entity model for entities extracted from chunks."""
    
    __tablename__ = "entities"
    
    id = Column(String, primary_key=True, index=True)
    text = Column(String, index=True)
    type = Column(String, index=True)
    chunk_id = Column(String, ForeignKey("chunks.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunk = relationship("Chunk", back_populates="entities")


class Query(Base):
    """Query model for user queries."""
    
    __tablename__ = "queries"
    
    id = Column(String, primary_key=True, index=True)
    text = Column(Text)
    enhanced_text = Column(Text, nullable=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String)
    parameters = Column(JSON, nullable=True)
    
    # Relationships
    responses = relationship("Response", back_populates="query")


class Response(Base):
    """Response model for RAG responses."""
    
    __tablename__ = "responses"
    
    id = Column(String, primary_key=True, index=True)
    query_id = Column(String, ForeignKey("queries.id"))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)  # List of chunk IDs used in context
    
    # Metrics
    relevance_score = Column(Float, nullable=True)
    truthfulness_score = Column(Float, nullable=True)
    completeness_score = Column(Float, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    estimated_cost = Column(Float, nullable=True)
    
    # Relationships
    query = relationship("Query", back_populates="responses")


class ApiKey(Base):
    """API key model for authentication."""
    
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True, index=True)
    key = Column(String, index=True, unique=True)
    name = Column(String)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    is_active = Column(Integer, default=1)  # Using Integer for SQLite compatibility 


class ProcessingStatusEnum(Enum):
    """Enum for document processing status values."""
    UPLOADED = "uploaded"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    GRAPH_BUILT = "graph_built"


class ProcessingStatus(Base):
    """Processing status model for tracking document processing."""
    
    __tablename__ = "processing_status"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), index=True)
    process_type = Column(String, index=True)  # chunking, embedding, graph, etc.
    status = Column(String, index=True)  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship (optional)
    document = relationship("Document") 