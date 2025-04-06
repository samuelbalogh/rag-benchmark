"""Models for the vector store service."""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field


@dataclass
class Document:
    """Document model for vector store."""
    
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            # Don't include embedding by default as it can be large
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            Document instance
        """
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            score=data.get("score"),
            embedding=data.get("embedding")
        ) 