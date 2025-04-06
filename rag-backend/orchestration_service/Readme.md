# Orchestration Service

## Overview

The Orchestration Service acts as the central coordinator for processing user queries within the RAG Benchmark Platform. It sits between the API Gateway and the various backend components (embedding, vector store, knowledge graph, LLM). Its primary purpose is to receive a user query along with the desired RAG configuration (strategy, models, enhancement techniques), execute the corresponding RAG pipeline, and return the generated answer along with relevant metadata and evaluation precursors.

Unlike the asynchronous `corpus_service` or `embedding_service` (which handle background document processing), the Orchestration Service typically operates **synchronously** within the context of an API request, as it needs to return a response to the user in a timely manner. It is likely invoked directly by the API Gateway's query endpoint handler.

## Responsibilities

*   **Request Parsing:** Interprets incoming query requests, extracting the user query, selected RAG strategy (e.g., Vector Search, Knowledge Graph, Hybrid), chosen models (embedding, LLM), and any specified query enhancement techniques.
*   **Query Enhancement (Optional):** Applies selected query enhancement strategies (e.g., synonym expansion, hypothetical document generation, LLM rewrite) to the user's query before retrieval. This might involve calling dedicated enhancement logic or services.
*   **Retrieval Coordination:**
    *   **Vector Search:** Generates an embedding for the (potentially enhanced) query using the selected embedding model (synchronously) and queries the vector store (e.g., `pgvector`) for relevant chunks.
    *   **Knowledge Graph:** Queries the knowledge graph store (e.g., loaded NetworkX graph) based on entities extracted from the (potentially enhanced) query.
    *   **Hybrid:** Executes both vector and graph retrieval and implements a strategy to fuse the results into a combined context.
*   **Context Formulation:** Assembles the retrieved information (chunks, graph paths) into a coherent context suitable for prompting an LLM.
*   **LLM Interaction:** Sends the formulated prompt (including the original query and retrieved context) to the chosen Large Language Model (e.g., OpenAI API) to generate an answer.
*   **Response Packaging:** Formats the LLM's response, potentially including source information (retrieved chunks/graph data), performance metrics (latency), and data needed for later evaluation.
*   **Logging:** Records details about the query, the chosen strategy, the retrieved context, the final response, and performance metrics for analysis and evaluation.

## Architecture & Technology

*   **Framework:** Likely integrated with or directly called by the FastAPI (`api_gateway`) application.
*   **Language:** Python
*   **Key Libraries:** Clients for LLM providers (e.g., `openai`), database connectors (`psycopg2` or ORM), potentially `networkx` for graph interaction, libraries for specific RAG techniques (e.g., Langchain if used).

## Interfaces & Communication

The Orchestration Service acts as a central hub, primarily communicating *synchronously* during a query request lifecycle.

1.  **Incoming Communication (API Layer Call):**
    *   **Mechanism:** Not a network endpoint itself, but rather its functions/classes are invoked directly by the API Gateway's request handler, specifically the one handling the `POST /api/v1/query` endpoint.
    *   **Caller:** `api_gateway.routers.query` module.
    *   **Payload (received from API Gateway):** User query string, selected RAG strategy identifier, parameters for the strategy (e.g., `k` for vector search, embedding model to use, LLM model to use), selected query enhancement options.

2.  **Outgoing Communication (Synchronous Calls/Interactions):**
    *   **Query Enhancement Logic:** May call internal functions or potentially a separate synchronous service/module dedicated to query enhancement.
    *   **Embedding Generation (for Query):** Needs to *synchronously* generate an embedding for the user's query. This might involve:
        *   Directly using embedding model client libraries (e.g., `openai.Embedding.create`).
        *   Calling a synchronous function within the `embedding_service` module (if structured to allow this). **Note:** It would *not* typically enqueue a Celery task here, as that's asynchronous.
    *   **Vector Store (Database):** Direct database connection (via `api_gateway`'s or its own connection pool) to PostgreSQL/`pgvector` to perform similarity searches.
    *   **Knowledge Graph Store:** Direct interaction with the graph data, likely involving loading a serialized graph file (e.g., NetworkX graph from `/data/graph_storage` if running in Docker context with shared volumes) and querying it.
    *   **LLM Service:** Direct API calls to the configured LLM provider (e.g., OpenAI API).
    *   **Database (Logging):** Writes logs about the query, steps taken, and results to a logging table in PostgreSQL.

3.  **Response:**
    *   **Mechanism:** Returns the final packaged response data (answer, sources, basic metrics) to its caller (the `api_gateway`'s query handler).
    *   **Recipient:** `api_gateway.routers.query`.

4.  **Shared Resources:**
    *   **Database (PostgreSQL):** Access for querying vector store and logging.
    *   **Configuration (`.env`):** Access to API keys (LLM, potentially embedding providers if called directly), database connection details, graph storage paths, default parameters.
    *   **Graph Storage (Filesystem):** Access to read serialized knowledge graph files if using file-based storage.

## Core Components (Internal)

*   `service.py`: Contains the primary logic for the orchestration flow, implementing the different RAG strategies and coordinating calls to other components/services.

## Workflow Example (Simplified Hybrid Strategy)

1.  `api_gateway` receives `POST /api/v1/query` request.
2.  `api_gateway` calls `orchestration_service.process_query(...)` function/method.
3.  Orchestrator applies selected Query Enhancement (e.g., rewrite with LLM).
4.  Orchestrator generates embedding for the enhanced query (synchronous call).
5.  Orchestrator queries `pgvector` using the embedding to get relevant chunks (Vector Retrieval).
6.  Orchestrator queries the loaded Knowledge Graph based on entities in the enhanced query (Graph Retrieval).
7.  Orchestrator fuses results from steps 5 & 6 into a combined context.
8.  Orchestrator formats a prompt with the original query and combined context.
9.  Orchestrator calls the selected LLM API with the prompt.
10. Orchestrator receives the LLM's generated answer.
11. Orchestrator packages the answer, source info (chunks/graph data), and metrics.
12. Orchestrator logs relevant data to the database.
13. Orchestrator returns the packaged response to the `api_gateway`.
14. `api_gateway` sends the HTTP response to the user.

## Configuration

Key environment variables impacting this service:

*   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
*   `OPENAI_API_KEY` (or keys for other configured LLM providers)
*   Potentially API keys for embedding models if called directly (`VOYAGE_API_KEY`, etc.)
*   `GRAPH_STORAGE_PATH` (if using file-based graph storage)
*   Default model names/parameters for RAG strategies.

Refer to `.env.example` in the project root for a more comprehensive list.