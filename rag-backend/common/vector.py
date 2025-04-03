import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2.extras import Json, execute_values
from sqlalchemy import text
from sqlalchemy.orm import Session

from common.logging import get_logger

logger = get_logger(__name__)


def create_embedding_record(
    db: Session, 
    chunk_id: uuid.UUID, 
    model_id: str, 
    embedding: List[float]
) -> uuid.UUID:
    """
    Create a new embedding record using pgvector.
    
    Args:
        db: Database session
        chunk_id: ID of the chunk
        model_id: ID of the embedding model
        embedding: Vector embedding as a list of floats
        
    Returns:
        UUID of the created embedding
    """
    # Convert regular Python list to a pgvector compatible string
    embedding_str = f"{{{','.join(str(x) for x in embedding)}}}"
    
    # Generate a UUID for the new embedding
    embedding_id = uuid.uuid4()
    
    # Execute raw SQL to insert embedding with the vector type
    db.execute(
        text("""
        INSERT INTO embeddings (id, chunk_id, model_id, embedding, created_at)
        VALUES (:id, :chunk_id, :model_id, :embedding::vector, NOW())
        """),
        {
            "id": embedding_id,
            "chunk_id": chunk_id,
            "model_id": model_id,
            "embedding": embedding_str,
        },
    )
    
    db.commit()
    
    return embedding_id


def search_similar_embeddings(
    db: Session,
    embedding: List[float],
    model_id: str,
    limit: int = 5,
    distance_threshold: Optional[float] = None,
    document_ids: Optional[List[uuid.UUID]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for similar embeddings using pgvector.
    
    Args:
        db: Database session
        embedding: Query embedding vector
        model_id: ID of the embedding model
        limit: Maximum number of results
        distance_threshold: Optional maximum distance threshold
        document_ids: Optional list of document IDs to filter by
        
    Returns:
        List of matching chunks with similarity scores
    """
    # Convert embedding to pgvector string
    embedding_str = f"{{{','.join(str(x) for x in embedding)}}}"
    
    # Build the query with optional filters
    query = """
    SELECT e.id, e.chunk_id, c.content, d.title, d.source, c.metadata,
           c.document_id, 1 - (e.embedding <=> :embedding::vector) as similarity
    FROM embeddings e
    JOIN chunks c ON e.chunk_id = c.id
    JOIN documents d ON c.document_id = d.id
    WHERE e.model_id = :model_id
    """
    
    params = {
        "embedding": embedding_str,
        "model_id": model_id,
    }
    
    # Add document_ids filter if provided
    if document_ids:
        document_ids_str = f"{{{','.join(str(x) for x in document_ids)}}}"
        query += " AND c.document_id = ANY(:document_ids)"
        params["document_ids"] = document_ids_str
        
    # Add distance threshold if provided
    if distance_threshold is not None:
        # Convert similarity threshold to distance threshold (1 - similarity)
        distance = 1.0 - distance_threshold
        query += f" AND (e.embedding <=> :embedding::vector) < {distance}"
        
    # Add ordering and limit
    query += " ORDER BY similarity DESC LIMIT :limit"
    params["limit"] = limit
    
    # Execute the query
    result = db.execute(text(query), params)
    
    # Process results
    matches = []
    for row in result:
        matches.append({
            "id": row.id,
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "content": row.content,
            "document_title": row.title,
            "source": row.source,
            "metadata": row.metadata,
            "similarity": float(row.similarity),
        })
    
    return matches


def batch_insert_embeddings(
    db: Session,
    embeddings: List[Tuple[uuid.UUID, uuid.UUID, str, List[float]]],
) -> None:
    """
    Insert multiple embeddings in a batch operation.
    
    Args:
        db: Database session
        embeddings: List of tuples (embedding_id, chunk_id, model_id, embedding_vector)
    """
    # Get raw connection from SQLAlchemy for efficient batch operations
    connection = db.connection().connection
    
    # Convert embeddings to PostgreSQL format
    rows = []
    for embedding_id, chunk_id, model_id, vector in embeddings:
        embedding_str = f"{{{','.join(str(x) for x in vector)}}}"
        rows.append((embedding_id, chunk_id, model_id, embedding_str))
    
    # Execute batch insert
    with connection.cursor() as cursor:
        execute_values(
            cursor,
            """
            INSERT INTO embeddings (id, chunk_id, model_id, embedding, created_at)
            VALUES %s
            """,
            [(id, chunk_id, model_id, f"{vector}::vector", "NOW()") for id, chunk_id, model_id, vector in rows],
        )
    
    db.commit() 