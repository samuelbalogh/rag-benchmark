# RAG Benchmark Backend

This repository contains the backend implementation for the RAG (Retrieval-Augmented Generation) Benchmark platform.

## Current Implementation Status

We have completed the following key components:

- **Document Processing Pipeline**
  - Multiple chunking strategies (fixed-length, semantic, paragraph-based)
  - Document metadata handling

- **Embedding Service**
  - Support for OpenAI embeddings (Ada, text-embedding-3-small, text-embedding-3-large)
  - Support for Voyage AI embeddings
  - Local embedding models using sentence-transformers
  - Embedding caching system

- **Vector Store**
  - PostgreSQL with pgvector for efficient vector storage and retrieval
  - Optimized vector search with HNSW and IVF-Flat indexes
  - Metadata filtering
  - Hybrid search combining dense and sparse retrieval
  - BM25 and TF-IDF implementations for keyword-based retrieval

- **Knowledge Graph**
  - Entity extraction using spaCy NER and custom patterns
  - Relationship extraction using dependency parsing
  - Graph storage and serialization with NetworkX
  - GraphML format for graph persistence

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Make (for using the Makefile)

### Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. Start the services with Docker Compose:
   ```bash
   make up
   ```

### Running Services

- **API Gateway**: 
  ```bash
  make api
  ```

- **Corpus Service Worker**:
  ```bash
  make corpus-worker
  ```

- **Development Mode** (auto-reload):
  ```bash
  make dev
  ```

## Architecture

The backend is designed as a collection of microservices:

- **API Gateway**: FastAPI application serving as the entry point
- **Corpus Service**: Handles document ingestion and chunking
- **Embedding Service**: Generates vector embeddings for document chunks
- **Vector Service**: Provides vector search capabilities
- **Knowledge Graph Service**: Builds and queries knowledge graphs
- **Query Service**: Processes user queries with different RAG strategies
- **Evaluation Service**: Measures performance of RAG implementations

## Database Schema

The system uses PostgreSQL with pgvector extension for vector storage. The schema includes:

- Documents
- Chunks
- Embeddings
- Query logs
- Response metrics
- Processing status
- Graph metadata

## Next Steps

The following areas are still under development:

- Complete query enhancement implementation
- RAG orchestration service
- Advanced evaluation metrics
- Full API endpoint implementation
- Integration with frontend
- Comprehensive testing suite

## License

[MIT License](LICENSE) 