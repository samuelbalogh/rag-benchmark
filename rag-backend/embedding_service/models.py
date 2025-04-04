"""Embedding models for vector representations."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os

import numpy as np
import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


class EmbeddingModel(ABC):
    """Base class for embedding models."""

    def __init__(self, model_id: str, dimensions: int):
        """
        Initialize embedding model.
        
        Args:
            model_id: Model identifier
            dimensions: Vector dimensions
        """
        self.model_id = model_id
        self.dimensions = dimensions
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    def get_batch_size(self) -> int:
        """
        Get batch size for embedding generation.
        
        Returns:
            Batch size
        """
        return 16


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model."""
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Batch requests to avoid token limits
            batch_size = self.get_batch_size()
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                response = openai_client.embeddings.create(
                    model=self.model_id,
                    input=batch_texts
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {str(e)}")
            raise


class VoyageEmbeddingModel(EmbeddingModel):
    """Voyage AI embedding model."""
    
    def __init__(self, model_id: str, dimensions: int):
        """
        Initialize Voyage embedding model.
        
        Args:
            model_id: Model identifier
            dimensions: Vector dimensions
        """
        super().__init__(model_id, dimensions)
        
        # Import voyage here to avoid dependency issues if not available
        try:
            import voyageai
            self.voyage_client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY", ""))
        except ImportError:
            logger.error("VoyageAI package not installed. Install with: pip install voyageai")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Voyage AI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Batch requests to avoid token limits
            batch_size = self.get_batch_size()
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                response = self.voyage_client.embed(
                    model=self.model_id,
                    input=batch_texts
                )
                
                all_embeddings.extend(response.embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Voyage AI: {str(e)}")
            raise


class LocalEmbeddingModel(EmbeddingModel):
    """Local embedding model using a local transformer model."""
    
    def __init__(self, model_id: str, dimensions: int):
        """
        Initialize local embedding model.
        
        Args:
            model_id: Model identifier (path to model)
            dimensions: Vector dimensions
        """
        super().__init__(model_id, dimensions)
        
        try:
            # Import sentence_transformers here to avoid dependency issues if not installed
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_id)
        except ImportError:
            logger.error("sentence_transformers package not installed. Install with: pip install sentence-transformers")
            raise
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using a local transformer model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Get embeddings
            embeddings = self.model.encode(texts)
            
            # Convert to Python list
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating embeddings with local model: {str(e)}")
            raise
    
    def get_batch_size(self) -> int:
        """Override batch size for local models."""
        return 8


def get_embedding_model(model_name: str) -> EmbeddingModel:
    """
    Get embedding model by name.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        EmbeddingModel instance
    """
    models = {
        # OpenAI models
        "text-embedding-ada-002": OpenAIEmbeddingModel("text-embedding-ada-002", 1536),
        "text-embedding-3-small": OpenAIEmbeddingModel("text-embedding-3-small", 1536),
        "text-embedding-3-large": OpenAIEmbeddingModel("text-embedding-3-large", 3072),
        
        # Voyage models
        "voyage-01": VoyageEmbeddingModel("voyage-01", 1024),
        "voyage-large-02": VoyageEmbeddingModel("voyage-large-02", 4096),
        
        # Local models (sentence-transformers)
        "all-MiniLM-L6-v2": LocalEmbeddingModel("all-MiniLM-L6-v2", 384),
        "all-mpnet-base-v2": LocalEmbeddingModel("all-mpnet-base-v2", 768),
    }
    
    if model_name not in models:
        logger.warning(f"Unknown embedding model: {model_name}, falling back to text-embedding-3-small")
        return models.get("text-embedding-3-small")
    
    return models.get(model_name) 