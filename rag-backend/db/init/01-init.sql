-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
  id UUID PRIMARY KEY,
  title TEXT NOT NULL,
  source TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Chunks table
CREATE TABLE chunks (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  metadata JSONB,
  chunk_index INTEGER NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Document processing status
CREATE TABLE processing_status (
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  process_type TEXT NOT NULL,  -- 'chunking', 'embedding', 'graph'
  status TEXT NOT NULL,        -- 'pending', 'processing', 'completed', 'failed'
  error_message TEXT,
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  PRIMARY KEY (document_id, process_type)
);

-- Embeddings table
CREATE TABLE embeddings (
  id UUID PRIMARY KEY,
  chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
  model_id TEXT NOT NULL,      -- 'ada', 'text-embedding-3-small', etc.
  embedding vector(1536),      -- dimension depends on model
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create an index for embeddings (HNSW for fast approximate search)
CREATE INDEX ON embeddings USING hnsw (embedding vector_l2_ops);

-- Query log
CREATE TABLE query_logs (
  id UUID PRIMARY KEY,
  query TEXT NOT NULL,
  enhanced_query TEXT,
  rag_strategy TEXT NOT NULL,  -- 'vector', 'graph', 'hybrid'
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Response metrics
CREATE TABLE response_metrics (
  id UUID PRIMARY KEY,
  query_id UUID REFERENCES query_logs(id) ON DELETE CASCADE,
  response TEXT NOT NULL,
  relevance_score FLOAT,
  truthfulness_score FLOAT,
  completeness_score FLOAT,
  latency_ms INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  estimated_cost FLOAT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- API keys for authentication
CREATE TABLE api_keys (
  id UUID PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Graph metadata table (to track graph versions)
CREATE TABLE graph_metadata (
  id UUID PRIMARY KEY,
  corpus_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  embedding_model TEXT NOT NULL,
  graph_type TEXT NOT NULL,  -- 'entity', 'semantic', 'hybrid'
  version INTEGER NOT NULL,
  node_count INTEGER NOT NULL,
  edge_count INTEGER NOT NULL,
  parameters JSONB,
  file_path TEXT NOT NULL,  -- Path to the serialized graph
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for frequent query patterns
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_model_id ON embeddings(model_id);
CREATE INDEX idx_processing_status_document_id ON processing_status(document_id);
CREATE INDEX idx_query_logs_rag_strategy ON query_logs(rag_strategy);
CREATE INDEX idx_response_metrics_query_id ON response_metrics(query_id); 