# rag benchmarking website - technical implementation plan

## 1. architecture overview

### 1.1 high-level architecture
- single page application (spa) with a python backend
- microservices architecture to separate rag implementations
- centralized evaluation service
- shared data processing pipeline

### 1.2 component breakdown
- **corpus processing service**: handles text ingestion, parsing, and preparation
- **embedding service**: manages different embedding models
- **vector store service**: interfaces with vector databases
- **knowledge graph service**: builds and queries knowledge graphs
- **query enhancement service**: implements different query reformulation techniques
- **rag orchestrator**: coordinates retrieval and generation across implementations
- **evaluation service**: computes metrics on responses
- **frontend application**: interactive user interface
- **api gateway**: unified interface for frontend to backend communication

## 2. data pipeline

### 2.1 corpus selection
- primary corpus: lord of the rings complete text
- secondary corpora: domain-specific texts (scientific papers, legal documents)
- reference corpus: curated qa pairs for evaluation

### 2.2 preprocessing workflow
- document segmentation strategies:
  - fixed-length chunks with overlap
  - semantic chunking based on document structure
  - paragraph-based chunking
- metadata extraction:
  - named entities
  - dates/locations
  - chapter/section markers
- document cleaning:
  - remove boilerplate
  - standardize formatting
  - handle special characters

### 2.3 embedding generation
- parallel processing of chunks across embedding models:
  - openai ada
  - openai text-embedding-3-small
  - openai text-embedding-3-large
  - voyage embeddings
- embedding versioning and caching system

### 2.4 knowledge graph construction
- entity extraction methods:
  - spacy ner
  - llm-based entity extraction
  - custom domain-specific extractors
- relationship mapping:
  - co-occurrence analysis
  - semantic relationship extraction
  - temporal and spatial relationship inference
- graph storage:
  - networkx for in-memory graph operations
  - json serialization for persistent storage in file system
  - graphml format for visualization and interoperability
  - property graph model with relationship types and strengths
  - nano-graphrag implementation for lightweight, flexible graph operations

## 3. backend implementation

### 3.1 core services
- **fastapi backend**: main api layer
- **celery workers**: async processing for long-running tasks
- **redis**: caching and message broker
- **postgresql**: relational data storage
- **vector databases**:
  - pgvector (postgresql extension)
  - qdrant or weaviate for specialized vector operations
  - nano-graphrag's built-in vector storage options (faiss, hnswlib, nano-vectordb)
- **graph storage**:
  - networkx for graph operations and algorithms
  - pickle or json serialization for persistence
  - stored in file system with versioning

### 3.2 rag implementations
- **baseline vector rag**:
  - simple similarity search
  - configurable k-nearest neighbors
  - optional metadata filtering
- **enhanced vector rag**:
  - hybrid sparse-dense retrieval
    - dense vectors for semantic similarity (embedding-based)
    - sparse vectors for keyword matching (bm25, tf-idf)
    - configurable weighting between sparse and dense scores
  - reranking with cross-encoders
  - relevance threshold filtering
- **knowledge graph rag**:
  - entity-centric retrieval
  - multi-hop traversal strategies
  - configurable path exploration depth
  - nano-graphrag for simplified, hackable graphrag implementation
- **hybrid rag**:
  - weighted combination of vector and graph retrieval
  - ensemble methods for context fusion
  - adaptive retrieval based on query type

### 3.3 query enhancement module
- **techniques**:
  - query expansion with synonyms
  - hypothetical document embeddings
  - query decomposition for complex questions
  - query transformation via llm
- **configuration interface**:
  - parameterization of enhancement strategies
  - a/b testing framework for strategy comparison

### 3.4 llm interaction layer
- openai api integration with configurable models
- response generation parameters:
  - temperature control
  - system prompt templates
  - context window management
  - optional response streaming

## 4. evaluation framework

### 4.1 metrics implementation
- **relevance**:
  - bertscore for semantic similarity
  - ragas relevance metrics
  - human evaluation interface
- **truthfulness**:
  - citation validation
  - factual consistency scoring
  - hallucination detection
- **completeness**:
  - answer coverage assessment
  - multi-perspective evaluation
- **performance**:
  - latency measurements
  - token usage tracking
  - cost calculation

### 4.2 visualization components
- real-time metric dashboards
- comparative analysis views
- performance over time tracking

## 5. frontend design

### 5.1 technology stack
- react with typescript
- tailwind css for styling
- react-query for data fetching
- d3.js for visualizations

### 5.2 key interface components
- **question exploration**:
  - curated question catalog
  - categorized by difficulty and type
  - custom question input
- **rag method selector**:
  - configuration panel for each method
  - parameter adjustment controls
- **response visualization**:
  - side-by-side comparison view
  - highlighted source passages
  - confidence indicators
- **evaluation dashboard**:
  - metric scorecard
  - spider/radar charts for multi-metric comparison
  - time-series performance graphs

### 5.3 interactive features
- document explorer to browse corpus
- knowledge graph visualization
- retrieval explainability view
- step-by-step rag pipeline visualization

## 6. implementation roadmap

### 6.1 phase 1: foundation
- corpus processing pipeline
- baseline vector rag implementation
- simple frontend with basic comparison
- initial evaluation metrics

### 6.2 phase 2: advanced features
- knowledge graph integration
- query enhancement techniques
- expanded evaluation framework
- interactive visualizations

### 6.3 phase 3: optimization
- performance tuning
- user experience refinement
- comprehensive test cases
- documentation

## 7. technical considerations

### 7.1 scalability
- containerization with docker
- horizontal scaling for embedding generation
- caching strategies for frequent queries

### 7.2 monitoring and logging
- structured logging with timestamps and service identifiers
- prometheus metrics collection
- api usage tracking
- error monitoring and alerting

### 7.3 deployment strategy
- backend deployment:
  - vercel fluid service for serverless backend hosting
  - managed through terraform for infrastructure as code
  - optimized for dynamic scaling and cold-start prevention
  - multi-region deployment for high availability
  - configured for in-function concurrency to optimize costs
- frontend deployment:
  - vercel for static site hosting with spa support
  - automatic ci/cd pipeline with github integration
- data storage:
  - vector databases hosted on digital ocean droplet
  - graph data (networkx) serialized and stored on digital ocean droplet
  - document corpus and embeddings cached in vercel edge network when possible
- fallback strategy:
  - complete application stack can be migrated to digital ocean droplet if needed
  - containerized services for easy migration between hosting providers
- ci/cd pipeline with github actions
- staging environment for testing

### 7.4 infrastructure as code
- terraform configuration for all infrastructure components
- vercel provider for managing:
  - fluid compute functions configuration
  - environment variables and secrets
  - domain configuration
  - project settings
- digital ocean provider for managing:
  - droplet resources
  - persistent volume configuration
  - networking and firewall settings
- version controlled infrastructure in a dedicated repository
- remote state management with terraform cloud or s3 backend
- environment-specific configurations (dev, staging, production)

### 7.5 sample terraform configuration
```hcl
# Provider configuration
terraform {
  required_providers {
    vercel = {
      source  = "vercel/vercel"
      version = "~> 0.15.0"
    }
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.30.0"
    }
  }
  backend "s3" {
    # Remote state configuration
  }
}

provider "vercel" {
  # API token from environment variable VERCEL_API_TOKEN
}

provider "digitalocean" {
  # Token from environment variable DIGITALOCEAN_TOKEN
}

# Vercel project configuration
resource "vercel_project" "rag_benchmark" {
  name      = "rag-benchmark"
  framework = "other"
  git_repository = {
    type = "github"
    repo = "username/rag-benchmark"
  }
  build_command   = "pip install -r requirements.txt && python build.py"
  output_directory = "dist"
  
  # Enable Fluid compute
  serverless_function_region = "iad1"
  serverless_function_config = {
    runtime                 = "python3.10"
    max_duration            = 60
    memory                  = 1024
    enable_fluid_compute    = true
    enable_concurrent_builds = true
  }
  
  environment = [
    {
      key    = "DATABASE_URL"
      value  = digitalocean_database_cluster.vector_db.uri
      target = ["production", "preview"]
    },
    {
      key    = "OPENAI_API_KEY"
      value  = var.openai_api_key
      target = ["production", "preview"]
    }
  ]
}

# Digital Ocean resources for persistent storage
resource "digitalocean_droplet" "backend_storage" {
  name     = "rag-benchmark-storage"
  region   = "nyc3"
  size     = "s-2vcpu-4gb"
  image    = "ubuntu-22-04-x64"
  ssh_keys = [digitalocean_ssh_key.terraform.id]
  
  connection {
    host        = self.ipv4_address
    user        = "root"
    type        = "ssh"
    private_key = file(var.pvt_key)
    timeout     = "2m"
  }
  
  provisioner "remote-exec" {
    inline = [
      "export PATH=$PATH:/usr/bin",
      "mkdir -p /data/graph_storage",
      "mkdir -p /data/vector_storage",
      "chmod -R 775 /data"
    ]
  }
}

# DNS Record for API
resource "vercel_project_domain" "api" {
  project_id = vercel_project.rag_benchmark.id
  domain     = "api.rag-benchmark.com"
}
```

## 8. tools and libraries

### 8.1 core dependencies
- langchain for rag orchestration
- huggingface transformers for embeddings
- spacy for nlp tasks
- ragas for evaluation
- networkx for graph operations and algorithms
- nano-graphrag for lightweight graphrag implementation
- pyterrier or rank_bm25 for sparse vector retrieval
- pytest for testing

### 8.2 development tools
- poetry for dependency management
- black and isort for code formatting
- mypy for type checking
- makefile for common tasks
- terraform for infrastructure management
- terraform-docs for documentation generation

## 9. extensibility considerations
- plugin architecture for new rag methods
- standardized interfaces for embedding models
- flexible evaluation metric addition
- corpus agnostic design

## 10. nano-graphrag integration

### 10.1 key features
- lightweight, hackable graphrag implementation (approximately 1100 lines of code)
- asynchronous operation support for better performance
- fully typed codebase for better development experience
- both local and global search modes
- incremental document insertion support

### 10.2 components utilization
- **built-in vector stores**: leveraging nano-graphrag's support for faiss, hnswlib, and nano-vectordb
- **embedding options**: utilizing multiple embedding models through nano-graphrag's flexible api
- **graph visualization**: using graphml output for knowledge graph visualization
- **chunking strategies**: implementing both token-based and text-splitter based chunking

### 10.3 implementation advantages
- simplified graphrag architecture compared to more complex implementations
- customizable prompts for entity extraction and community reporting
- lower computational overhead for faster response times
- easy integration with various llm providers (openai, bedrock, ollama, etc.)
- support for batch and incremental document processing

## 11. networkx knowledge graph implementation

### 11.1 graph data model
- entities stored as nodes with properties for metadata
- relationships stored as edges with weight and type attributes
- hierarchical relationships represented through directed edges
- community structure identified through networkx's community detection algorithms

### 11.2 storage architecture
- in-memory graph representation during processing
- persistent storage using json serialization for human readability
- pickle serialization for larger graphs with better performance
- versioned storage with timestamped files for incremental updates
- file path structure: `data/graphs/{corpus_id}/{embedding_model}/{timestamp}.{format}`

### 11.3 graph operations
- entity resolution through similarity measures and custom matching rules
- relationship inference using co-occurrence and semantic similarity
- multi-hop traversal for complex question answering
- community detection for topic clustering
- centrality measures for identifying key entities

### 11.4 integration with other components
- vector search integration for hybrid retrieval
- llm-guided traversal for complex queries
- visualization export in graphml format for interactive exploration
- metrics collection on graph statistics (density, clustering, path lengths)
- bidirectional integration with nano-graphrag for algorithm comparison
