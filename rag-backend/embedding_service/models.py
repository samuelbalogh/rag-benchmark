"""Embedding models for generating vector representations."""

import numpy as np


class EmbeddingModel:
    """Model for generating text embeddings."""
    
    def __init__(self, model_id, dimensions):
        """Initialize embedding model.
        
        Args:
            model_id: Identifier for the model
            dimensions: Dimensions of the embeddings
        """
        self.model_id = model_id
        self.dimensions = dimensions
    
    def embed(self, text):
        """Generate embedding for text.
        
        For the test environment, this generates a random vector.
        In production, it would call an embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # In production, this would call the OpenAI API or other embedding provider
        # For testing, we return a random vector of the correct dimensions
        return np.random.rand(self.dimensions).tolist()
    
    def embed_batch(self, texts):
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts] 