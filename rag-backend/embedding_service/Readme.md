# Embedding Service

## Overview

The Embedding Service is a specialized backend component within the RAG Benchmark Platform responsible for generating vector representations (embeddings) of text chunks. These embeddings are fundamental for semantic search and retrieval in various RAG strategies. This service supports multiple embedding models (e.g., OpenAI Ada, text-embedding-3-small/large, Voyage) as outlined in the project goals, allowing for comparison of their effectiveness.

Like the Corpus Service, it operates asynchronously, typically as a Celery worker, to handle the potentially computationally intensive process of interacting with embedding model APIs or libraries without blocking other operations.

## Responsibilities

*   **Embedding Generation:** Receives requests to generate vector embeddings for specified text chunks using a designated embedding model.
*   **Model Interaction:** Interfaces with various embedding model providers (e.g., OpenAI API, Voyage AI API, local Hugging Face models).
*   **Vector Storage:** Persists the generated embeddings into the configured vector store (e.g., `pgvector` within the PostgreSQL database).
*   **Status Updates:** Updates the status of chunks in the main database to indicate that embeddings have been successfully generated (or if generation failed).
*   **Configuration Management:** Loads and utilizes necessary API keys and model identifiers from environment variables or a configuration source.

## Architecture & Technology

*   **Framework:** Likely Celery (based on overall architecture)
*   **Message Broker:** Redis (for Celery)
*   **Language:** Python
*   **Vector Storage:** PostgreSQL with `pgvector` extension (as per `docker-compose.yml`)
*   **Key Libraries:** Clients for embedding providers (e.g., `openai`), database connectors (e.g., `psycopg2` or an ORM like SQLAlchemy), Celery.

## Interfaces & Communication

The Embedding Service primarily interacts asynchronously via task queues and directly with shared data stores. It **does not typically expose direct HTTP API endpoints**.

1.  **Incoming Communication (Task Queue):**
    *   **Mechanism:** Consumes Celery tasks from the Redis queue.
    *   **Producer:** The `corpus_service` is the most likely producer. After successfully chunking a document, it would enqueue tasks for the `embedding_service` to generate embeddings for those chunks, potentially specifying which embedding model(s) to use.
    *   **Example Task (Conceptual):** `embedding_service.tasks.generate_embeddings(chunk_ids=[<uuid1>, <uuid2>, ...], model_name='text-embedding-3-small')` or perhaps `embedding_service.tasks.generate_embedding_for_chunk(chunk_id=<uuid>, model_name='voyage-02')`.

2.  **Outgoing Communication (Direct Database Interaction):**
    *   **Mechanism:** Direct connection to the PostgreSQL database.
    *   **Actions:**
        *   Writing the generated vector embeddings to the appropriate table(s) managed by `pgvector`.
        *   Updating metadata tables (e.g., a `chunks` table) to mark the embedding status as `COMPLETED` or `FAILED` for the specific chunk and model combination.

3.  **Shared Resources:**
    *   **Database (PostgreSQL with pgvector):** Reads chunk text content (or retrieves it via chunk ID) and writes embedding vectors and status updates.
    *   **Task Queue (Redis):** Consumes tasks enqueued by other services.
    *   **Configuration (`.env`):** Relies on environment variables for database connection details, Redis URL, API keys for embedding models (e.g., `OPENAI_API_KEY`, `VOYAGE_API_KEY`), and potentially default model choices.

## Core Components (Internal)

*   `service.py`: Likely contains the main logic for interacting with embedding APIs/libraries, handling task execution (if using Celery), and database interactions. May define the Celery worker application.
*   `models.py`: Potentially defines data structures or Pydantic models for representing embedding configurations, chunk data, or API responses.

## Running Locally (Development)

Assuming it runs as a Celery worker (adjust `-A` parameter if the Celery app instance is defined elsewhere, e.g., in `service.py`):

1.  Ensure all necessary environment variables are set (API keys, DB/Redis connection strings).
2.  Navigate to the `rag-backend` directory.
3.  Run the Celery worker:
    ```bash
    # Adjust 'embedding_service.worker' if the Celery app is defined elsewhere
    celery -A embedding_service.worker worker --loglevel=info 
    ```

## Configuration

Key environment variables required:

*   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
*   `REDIS_HOST`, `REDIS_PORT` (or `CELERY_BROKER_URL`)
*   `OPENAI_API_KEY` (if using OpenAI models)
*   `VOYAGE_API_KEY` (if using Voyage models)
*   Potentially others depending on chosen embedding models.

Refer to `.env.example` in the project root for a more comprehensive list.