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
  
- **Query Enhancement**
  - Synonym expansion using NLTK WordNet
  - Hypothetical document generation with LLMs
  - Query decomposition for complex queries
  - LLM-based query rewriting for better retrieval
  - Combined enhancement methods

- **RAG Strategies**
  - Vector search strategy with embedding-based retrieval
  - Knowledge graph strategy for entity-based retrieval
  - Hybrid strategy combining vector and graph approaches
  - Strategy factory pattern for extensibility

- **RAG Orchestration Service**
  - Coordination of multiple RAG components
  - Predefined strategy configurations (vector, knowledge graph, hybrid, enhanced vector)
  - Strategy comparison functionality
  - Adaptive query processing based on query analysis
  - Custom parameter overrides for fine-tuning

- **Evaluation Service**
  - Multiple evaluation metrics for RAG responses
  - Relevance assessment using TF-IDF similarity
  - Faithfulness evaluation comparing responses to source documents
  - Context recall measurement against ground truth
  - LLM-based answer relevancy scoring

- **API Gateway**
  - RESTful API endpoints for query processing
  - Strategy selection and comparison
  - Adaptive query processing
  - Health check and monitoring

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

### Testing the Query Service

We provide a test script to validate the query service implementation:

```bash
python test_query_service.py
```

You can test specific components with:

```bash
python test_query_service.py enhancement  # Test query enhancement methods
python test_query_service.py strategies    # Test retrieval strategies
python test_query_service.py full          # Test full query processing
```

### Testing the Orchestration Service

To test the RAG orchestration service:

```bash
python test_orchestration.py
```

You can test specific aspects with:

```bash
python test_orchestration.py strategies    # Test different RAG strategies
python test_orchestration.py comparison    # Test strategy comparison
python test_orchestration.py complex       # Test complex query processing
python test_orchestration.py custom        # Test custom strategy parameters
```

### Testing the Evaluation Service

To evaluate the performance of RAG responses:

```bash
python test_evaluation.py
```

You can test specific evaluation metrics with:

```bash
python test_evaluation.py individual    # Test individual metrics
python test_evaluation.py multiple      # Test multiple evaluation methods
python test_evaluation.py real          # Test with real RAG output
```

### Running All Tests

To run all tests in one go:

```bash
python test_all.py
```

You can also run a specific test suite:

```bash
python test_all.py test_query_service
```

Or a specific test within a suite:

```bash
python test_all.py test_orchestration.complex
```

## Using the API

Once the API is running, you can access it at http://localhost:8000. The following endpoints are available:

- **Process Query**: `POST /query/process`
  - Process a query using a specific strategy
  - Example request:
    ```json
    {
      "query": "What are the main components of a RAG system?",
      "strategy_key": "vector",
      "return_documents": true
    }
    ```

- **Compare Strategies**: `POST /query/compare`
  - Compare multiple strategies for a query
  - Example request:
    ```json
    {
      "query": "Explain the difference between vector and hybrid search.",
      "strategy_keys": ["vector", "hybrid", "knowledge_graph"]
    }
    ```

- **Adaptive Query Processing**: `POST /query/adaptive`
  - Process a query using adaptive strategy selection
  - Example request:
    ```json
    {
      "query": "Who created transformer models and when were they first introduced?",
      "return_documents": true
    }
    ```

Interactive API documentation is available at http://localhost:8000/docs

## Architecture

The backend is designed as a collection of microservices:

- **API Gateway**: FastAPI application serving as the entry point
- **Corpus Service**: Handles document ingestion and chunking
- **Embedding Service**: Generates vector embeddings for document chunks
- **Vector Service**: Provides vector search capabilities
- **Knowledge Graph Service**: Builds and queries knowledge graphs
- **Query Service**: Processes user queries with different RAG strategies
- **Orchestration Service**: Coordinates RAG components and strategy selection
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

- Advanced evaluation metrics
- Full API endpoint implementation
- Integration with frontend
- Comprehensive testing suite
- Horizontal scaling for high-volume processing

## License

[MIT License](LICENSE) 