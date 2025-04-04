import uuid
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import nltk
from nltk.tokenize import sent_tokenize, TextTilingTokenizer
from sklearn.cluster import AgglomerativeClustering

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/texttiling')
except LookupError:
    nltk.download('stopwords')


class ChunkingStrategy(ABC):
    """Base class for document chunking strategies."""

    @abstractmethod
    def chunk_document(self, content: str, document_id: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller segments.
        
        Args:
            content: Document content
            document_id: ID of the document
            **kwargs: Additional parameters for specific chunking strategies
            
        Returns:
            List of chunks
        """
        pass


class FixedLengthChunking(ChunkingStrategy):
    """Chunking strategy that splits content into fixed-length segments with overlap."""
    
    def chunk_document(
        self, 
        content: str, 
        document_id: str, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into fixed-length segments.
        
        Args:
            content: Document content
            document_id: ID of the document
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # If content is small enough, use as a single chunk
        if len(content) <= chunk_size:
            chunks.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": content,
                "metadata": {"index": 0, "start": 0, "end": len(content), "strategy": "fixed_length"},
                "chunk_index": 0,
            })
            return chunks
        
        # Otherwise, chunk with overlap
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position
            end = min(start + chunk_size, len(content))
            
            # If not at the end of the content and not the first chunk
            # Try to find a good break point (period followed by space)
            if end < len(content) and start > 0:
                # Look for the last period in the chunk
                last_period = content.rfind('. ', start, end)
                if last_period > start + chunk_size // 2:  # Only use if it's in the latter half
                    end = last_period + 1  # Include the period
            
            # Create chunk
            chunk_content = content[start:end]
            
            # Add to chunks list
            chunks.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk_content,
                "metadata": {
                    "index": chunk_index, 
                    "start": start, 
                    "end": end, 
                    "strategy": "fixed_length"
                },
                "chunk_index": chunk_index,
            })
            
            # Update start position for next chunk
            start = end - chunk_overlap if end < len(content) else len(content)
            chunk_index += 1
        
        return chunks


class ParagraphChunking(ChunkingStrategy):
    """Chunking strategy that splits content by paragraphs."""
    
    def chunk_document(
        self, 
        content: str, 
        document_id: str, 
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document by paragraphs.
        
        Args:
            content: Document content
            document_id: ID of the document
            max_chunk_size: Maximum size of a chunk
            min_chunk_size: Minimum size of a chunk
            
        Returns:
            List of chunks
        """
        # Split content by paragraph breaks (empty lines)
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for i, para in enumerate(paragraphs):
            para_size = len(para)
            
            # If adding this paragraph would exceed max_chunk_size and we have content,
            # or if this is the last paragraph, create a chunk
            if ((current_size + para_size > max_chunk_size and current_chunk) or
                (i == len(paragraphs) - 1)):
                
                # Add current paragraph if this is the last one
                if i == len(paragraphs) - 1:
                    current_chunk.append(para)
                    current_size += para_size
                
                # Join paragraphs into a single chunk
                chunk_content = "\n\n".join(current_chunk)
                
                # Add to chunks list
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": chunk_content,
                    "metadata": {
                        "index": chunk_index, 
                        "size": current_size,
                        "paragraphs": len(current_chunk),
                        "strategy": "paragraph"
                    },
                    "chunk_index": chunk_index,
                })
                
                # Reset for next chunk
                current_chunk = [para] if i < len(paragraphs) - 1 else []
                current_size = para_size if i < len(paragraphs) - 1 else 0
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_size += para_size
        
        # If we have a partial chunk left, add it
        if current_chunk and current_size >= min_chunk_size:
            chunk_content = "\n\n".join(current_chunk)
            chunks.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk_content,
                "metadata": {
                    "index": chunk_index, 
                    "size": current_size,
                    "paragraphs": len(current_chunk),
                    "strategy": "paragraph"
                },
                "chunk_index": chunk_index,
            })
        
        return chunks


class SemanticChunking(ChunkingStrategy):
    """Chunking strategy that splits content based on semantic similarity."""
    
    def chunk_document(
        self, 
        content: str, 
        document_id: str, 
        max_chunk_size: int = 2000,
        min_sentences: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document based on semantic boundaries using TextTiling.
        
        Args:
            content: Document content
            document_id: ID of the document
            max_chunk_size: Maximum size of a chunk
            min_sentences: Minimum number of sentences in a chunk
            
        Returns:
            List of chunks
        """
        # Split into sentences
        sentences = sent_tokenize(content)
        
        # If very short document, return as single chunk
        if len(sentences) <= min_sentences:
            return [{
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": content,
                "metadata": {
                    "index": 0, 
                    "sentences": len(sentences),
                    "strategy": "semantic"
                },
                "chunk_index": 0,
            }]
        
        try:
            # Use TextTiling algorithm to find semantic boundaries
            tt = TextTilingTokenizer(w=15, k=10)
            tile_boundaries = tt.tokenize(content)
            
            # Create chunks from tiles
            chunks = []
            for i, tile in enumerate(tile_boundaries):
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": tile,
                    "metadata": {
                        "index": i,
                        "sentences": len(sent_tokenize(tile)),
                        "strategy": "semantic_texttiling"
                    },
                    "chunk_index": i,
                })
            
            return chunks
        except Exception as e:
            # Fallback to simpler sentence-based chunking if TextTiling fails
            return self._sentence_based_chunking(sentences, document_id, max_chunk_size, min_sentences)
    
    def _sentence_based_chunking(
        self, 
        sentences: List[str], 
        document_id: str, 
        max_chunk_size: int,
        min_sentences: int
    ) -> List[Dict[str, Any]]:
        """Fallback chunking method based on sentences."""
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            sent_size = len(sentence)
            
            # If adding this sentence would exceed max_chunk_size and we have enough sentences,
            # or if this is the last sentence, create a chunk
            if ((current_size + sent_size > max_chunk_size and len(current_chunk) >= min_sentences) or
                (i == len(sentences) - 1)):
                
                # Add current sentence if this is the last one
                if i == len(sentences) - 1:
                    current_chunk.append(sentence)
                
                # Join sentences into a single chunk
                chunk_content = " ".join(current_chunk)
                
                # Add to chunks list
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "content": chunk_content,
                    "metadata": {
                        "index": chunk_index, 
                        "sentences": len(current_chunk),
                        "strategy": "semantic_sentence"
                    },
                    "chunk_index": chunk_index,
                })
                
                # Reset for next chunk
                current_chunk = [sentence] if i < len(sentences) - 1 else []
                current_size = sent_size if i < len(sentences) - 1 else 0
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sent_size
        
        # If we have a partial chunk left, add it
        if current_chunk and len(current_chunk) >= min_sentences:
            chunk_content = " ".join(current_chunk)
            chunks.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk_content,
                "metadata": {
                    "index": chunk_index, 
                    "sentences": len(current_chunk),
                    "strategy": "semantic_sentence"
                },
                "chunk_index": chunk_index,
            })
        
        return chunks


def get_chunking_strategy(strategy_name: str) -> ChunkingStrategy:
    """
    Get chunking strategy by name.
    
    Args:
        strategy_name: Name of the chunking strategy
        
    Returns:
        ChunkingStrategy instance
    """
    strategies = {
        "fixed_length": FixedLengthChunking(),
        "paragraph": ParagraphChunking(),
        "semantic": SemanticChunking(),
    }
    
    return strategies.get(strategy_name, FixedLengthChunking()) 