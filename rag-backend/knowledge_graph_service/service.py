"""Knowledge graph service for entity extraction and relationship mapping."""

import logging
import time
import os
import json
import uuid
from typing import Dict, List, Any, Optional

import networkx as nx
from sqlalchemy.orm import Session

from common.database import SessionLocal
from common.logging import get_logger
from common.models import Document, Chunk, ProcessingStatus
from knowledge_graph_service.entity_extraction import (
    extract_entities,
    extract_relations,
    build_knowledge_graph
)

# Initialize logger
logger = get_logger(__name__)

# Create graphs directory if it doesn't exist
GRAPH_DIR = os.environ.get("GRAPH_STORAGE_PATH", "./data/graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


async def process_document_for_graph(document_id: str) -> Dict[str, Any]:
    """
    Process a document to create a knowledge graph.
    
    Args:
        document_id: ID of the document
        
    Returns:
        Dictionary with processing status
    """
    logger.info(f"Processing document for knowledge graph: {document_id}")
    start_time = time.time()
    
    # Create database session
    db = SessionLocal()
    
    try:
        # Update processing status
        update_processing_status(db, document_id, "graph", "processing")
        
        # Get document and chunks
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document not found: {document_id}")
            return {"success": False, "error": "Document not found"}
        
        chunks = db.query(Chunk).filter(Chunk.document_id == document_id).all()
        if not chunks:
            logger.warning(f"No chunks found for document: {document_id}")
            update_processing_status(db, document_id, "graph", "failed", "No chunks found")
            return {"success": False, "error": "No chunks found"}
        
        # Process each chunk for entities
        all_entities = []
        all_relations = []
        
        for chunk in chunks:
            # Extract entities
            entities = extract_entities(chunk.content, str(chunk.id))
            all_entities.extend(entities)
            
            # Extract relationships
            relations = extract_relations(chunk.content, entities)
            all_relations.extend(relations)
        
        # Build knowledge graph
        graph = build_knowledge_graph(all_entities, all_relations)
        
        # Save graph
        graph_file = os.path.join(GRAPH_DIR, f"{document_id}.graphml")
        nx.write_graphml(graph, graph_file)
        
        # Update graph metadata in database
        metadata = {
            "id": str(uuid.uuid4()),
            "corpus_id": document_id,
            "graph_type": "entity",
            "version": 1,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "file_path": graph_file,
            "entities_count": len(all_entities),
            "relations_count": len(all_relations),
            "entity_types": list(set(entity["type"] for entity in all_entities)),
            "relation_types": list(set(rel["type"] for rel in all_relations))
        }
        
        # Update document metadata with graph information
        if document.metadata:
            document.metadata.update({
                "graph": metadata
            })
        else:
            document.metadata = {
                "graph": metadata
            }
        db.commit()
        
        # Save entities and relations for later use
        entities_file = os.path.join(GRAPH_DIR, f"{document_id}_entities.json")
        relations_file = os.path.join(GRAPH_DIR, f"{document_id}_relations.json")
        
        with open(entities_file, 'w') as f:
            json.dump(all_entities, f)
        
        with open(relations_file, 'w') as f:
            json.dump(all_relations, f)
        
        # Update processing status
        update_processing_status(db, document_id, "graph", "completed")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        logger.info(f"Knowledge graph created in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "document_id": document_id,
            "nodes_count": len(graph.nodes),
            "edges_count": len(graph.edges),
            "entities_count": len(all_entities),
            "relations_count": len(all_relations),
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Error processing document for graph: {str(e)}", exc_info=True)
        update_processing_status(db, document_id, "graph", "failed", str(e))
        return {"success": False, "error": str(e)}
    finally:
        db.close()


def update_processing_status(
    db: Session,
    document_id: str,
    process_type: str,
    status: str,
    error_message: Optional[str] = None
) -> None:
    """
    Update document processing status.
    
    Args:
        db: Database session
        document_id: ID of the document
        process_type: Type of processing (graph, embedding, etc.)
        status: Status to set
        error_message: Optional error message
    """
    try:
        # Get processing status
        processing_status = db.query(ProcessingStatus).filter(
            ProcessingStatus.document_id == document_id,
            ProcessingStatus.process_type == process_type
        ).first()
        
        if processing_status:
            # Update status
            processing_status.status = status
            processing_status.error_message = error_message
        else:
            # Create new status
            processing_status = ProcessingStatus(
                document_id=document_id,
                process_type=process_type,
                status=status,
                error_message=error_message
            )
            db.add(processing_status)
        
        # Commit changes
        db.commit()
    except Exception as e:
        logger.error(f"Error updating processing status: {str(e)}", exc_info=True)
        db.rollback()


def get_graph(document_id: str) -> Optional[nx.Graph]:
    """
    Get knowledge graph for a document.
    
    Args:
        document_id: ID of the document
        
    Returns:
        NetworkX graph or None if not found
    """
    graph_file = os.path.join(GRAPH_DIR, f"{document_id}.graphml")
    
    if os.path.exists(graph_file):
        try:
            return nx.read_graphml(graph_file)
        except Exception as e:
            logger.error(f"Error reading graph file: {str(e)}", exc_info=True)
            return None
    else:
        logger.warning(f"Graph file not found for document: {document_id}")
        return None 