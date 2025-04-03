"""Entity extraction for knowledge graph generation."""

import uuid


def extract_entities(text, chunk_id=None):
    """Extract entities from text.
    
    Args:
        text: Text to process
        chunk_id: ID of the chunk (optional)
        
    Returns:
        List of extracted entities
    """
    # This is a placeholder for spaCy or other NER system
    # In production, this would use a proper NLP pipeline
    
    # Sample entities for testing
    entities = []
    
    if "Alice" in text:
        entities.append({
            "id": f"e{uuid.uuid4().hex[:8]}",
            "text": "Alice",
            "type": "PERSON",
            "chunk_id": chunk_id
        })
    
    if "Bob" in text:
        entities.append({
            "id": f"e{uuid.uuid4().hex[:8]}",
            "text": "Bob",
            "type": "PERSON",
            "chunk_id": chunk_id
        })
    
    if "London" in text:
        entities.append({
            "id": f"e{uuid.uuid4().hex[:8]}",
            "text": "London",
            "type": "LOCATION",
            "chunk_id": chunk_id
        })
    
    if "Google" in text:
        entities.append({
            "id": f"e{uuid.uuid4().hex[:8]}",
            "text": "Google",
            "type": "ORGANIZATION",
            "chunk_id": chunk_id
        })
    
    return entities


# Placeholder for spaCy NLP pipeline
spacy_nlp = lambda text: None 