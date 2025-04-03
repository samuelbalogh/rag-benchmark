# rag benchmark platform - backend specification

## overview

the rag benchmark platform is designed to provide a comprehensive environment for evaluating and comparing different retrieval-augmented generation (rag) strategies. the platform supports various embedding models, retrieval methods, and evaluation metrics.

## architecture

### components

1. **api gateway service**
   - handles http requests
   - manages authentication
   - routes requests to appropriate services
   - provides unified api for frontend applications

2. **corpus service**
   - processes uploaded documents
   - handles document chunking
   - manages document storage
   - triggers embedding generation

3. **embedding service**
   - generates vector embeddings for document chunks
   - supports multiple embedding models
   - handles vector storage and indexing

4. **graph service**
   - builds knowledge graphs from documents
   - provides graph-based retrieval
   - supports entity extraction and relationship mapping

5. **query service**
   - processes user queries
   - implements various rag strategies
   - retrieves relevant context
   - calls llm apis for response generation

6. **evaluation service**
   - calculates performance metrics
   - runs benchmarks
   - tracks experiment results
   - provides comparative analysis

### technologies

- **backend framework**: fastapi
- **database**: postgresql with pgvector extension
- **task queue**: celery with redis
- **containerization**: docker and docker-compose
- **authentication**: api key-based auth
- **vector storage**: pgvector for postgres
- **logging**: structured json logs

## data models

### core entities

1. **document**
   - source documents uploaded by users
   - stored with metadata and processing status

2. **chunk**
   - segments of documents used for retrieval
   - linked to original documents
   - stored with positional metadata

3. **embedding**
   - vector representations of chunks
   - linked to embedding models
   - used for similarity search

4. **processing status**
   - tracks document processing stages
   - includes chunking, embedding, and graph creation

5. **query logs**
   - records user queries
   - stores rag strategy used
   - links to response metrics

6. **response metrics**
   - stores evaluation metrics for responses
   - includes relevance, truthfulness, completeness scores
   - tracks latency and token usage

7. **graph metadata**
   - stores knowledge graph information
   - includes node/edge counts and graph parameters

## api endpoints

### document management

- `POST /api/v1/documents`: upload new document
- `GET /api/v1/documents`: list documents
- `GET /api/v1/documents/{document_id}`: get document details
- `GET /api/v1/documents/{document_id}/status`: check processing status
- `DELETE /api/v1/documents/{document_id}`: delete document

### query processing

- `POST /api/v1/query`: process rag query
- `GET /api/v1/query/history`: get query history
- `POST /api/v1/query/benchmark`: run benchmark with preset questions

### configuration

- `GET /api/v1/config/embedding-models`: list available embedding models
- `GET /api/v1/config/rag-strategies`: list available rag strategies
- `GET /api/v1/config/query-enhancement-strategies`: list query enhancement options

### admin

- `GET /api/v1/admin/stats`: get system statistics
- `POST /api/v1/admin/api-keys`: create new api key
- `GET /api/v1/admin/api-keys`: list api keys
- `DELETE /api/v1/admin/api-keys/{key_id}`: delete api key
- `POST /api/v1/admin/rebuild-indexes`: rebuild vector indexes
- `POST /api/v1/admin/rebuild-graph`: rebuild knowledge graph

## rag strategies

the platform supports three main retrieval strategies:

1. **vector search** (default)
   - pure vector similarity search
   - configurable chunk count and similarity threshold
   - fastest but less contextually aware

2. **knowledge graph**
   - entity-centric retrieval using knowledge graph
   - supports multi-hop queries
   - better for relationship-focused questions

3. **hybrid**
   - combines vector search with knowledge graph
   - weighted fusion of results
   - best overall performance for complex queries

## query enhancement

the platform supports query enhancement techniques:

1. **synonym expansion**
   - expands query with synonyms of key terms

2. **hypothetical document**
   - generates hypothetical perfect document for the query

3. **query decomposition**
   - breaks complex queries into simpler subqueries

4. **llm rewrite**
   - uses llm to rephrase query for better retrieval

## evaluation metrics

the platform tracks multiple performance metrics:

1. **relevance score**
   - measures how relevant the retrieved context is to the query

2. **truthfulness score**
   - assesses factual accuracy of the response

3. **completeness score**
   - evaluates how completely the response addresses the query

4. **latency**
   - measures end-to-end response time

5. **token usage**
   - counts tokens used for context and response

6. **estimated cost**
   - calculates api usage costs

## hosting and deployment

### hosting solutions

1. **vercel fluid compute**
   - primary hosting platform for backend services
   - serverless functions with improved cold start performance
   - optimized for dynamic scaling based on load
   - in-function concurrency for cost optimization
   - multi-region deployment capability
   - configured via terraform for infrastructure as code
   - api endpoints accessible via custom domain (api.rag-benchmark.com)

2. **digital ocean**
   - persistent storage for vector databases and graph data
   - dedicated droplet for storage operations
   - configured storage paths:
     - `/data/graph_storage`: serialized networkx graphs
     - `/data/vector_storage`: vector embeddings and indexes
   - fallback hosting option for complete application stack
   - resource provisioning managed via terraform

3. **vercel (frontend)**
   - static site hosting for single page application
   - integrated with github for continuous deployment
   - edge caching for improved performance

### deployment architecture

- microservices deployed as serverless functions on vercel fluid
- persistent data storage on digital ocean droplet
- edge caching for frequently accessed documents and embeddings
- full containerization for easy migration between hosting providers
- terraform for infrastructure as code management
- environment-specific configurations (dev, staging, production)
- automatic ci/cd pipeline through github actions

## deployment

### prerequisites

- docker and docker-compose
- postgres with pgvector extension
- redis
- openai api key (for embeddings and llm)

### environment variables

key environment variables:

```
# api settings
API_HOST=0.0.0.0
API_PORT=8000

# database settings
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_benchmark

# redis settings
REDIS_HOST=localhost
REDIS_PORT=6379

# celery settings
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# openai api settings
OPENAI_API_KEY=your_openai_api_key

# document storage
DOCUMENT_STORAGE_PATH=/path/to/document/storage

# security
SECRET_KEY=your_secret_key_here
API_KEY_HEADER=X-API-Key

# vercel fluid settings
VERCEL_REGION=iad1
VERCEL_MAX_DURATION=60
VERCEL_MEMORY=1024

# digital ocean settings
DO_DROPLET_REGION=nyc3
DO_DROPLET_SIZE=s-2vcpu-4gb
DO_STORAGE_PATH=/data
```

### deployment steps

1. clone repository
2. copy `.env.example` to `.env` and configure
3. run `docker-compose up -d` to start all services
4. access api at `http://localhost:8000`
5. flower dashboard at `http://localhost:5555`

## development

### local setup

1. install poetry
2. run `poetry install` to install dependencies
3. set up postgres with pgvector
4. set up redis
5. configure environment variables
6. run fastapi server with `uvicorn api_gateway.main:app --reload`
7. run celery worker with `celery -A corpus_service.worker worker --loglevel=info`

### testing

- unit tests with pytest
- integration tests with docker-compose test setup
- benchmark tests for performance evaluation 