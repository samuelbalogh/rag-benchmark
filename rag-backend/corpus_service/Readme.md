# Corpus Service

## Overview

The Corpus Service is a crucial backend component of the RAG Benchmark Platform. Its primary responsibility is to handle the ingestion, processing, and preparation of source documents that will be used for Retrieval-Augmented Generation (RAG). It operates as an asynchronous Celery worker, decoupling time-consuming document processing tasks from the main API request/response cycle.

This service takes raw uploaded documents, breaks them down into manageable chunks using various strategies, stores them, and potentially triggers subsequent processing steps like embedding generation or knowledge graph construction.

## Responsibilities

*   **Document Ingestion:** Receives tasks to process newly uploaded documents.
*   **Preprocessing & Cleaning:** Performs necessary cleaning steps on the document text (e.g., removing boilerplate, standardizing formatting).
*   **Chunking:** Segments documents into smaller chunks based on configured strategies (e.g., fixed-size, semantic, paragraph-based). See `chunking_strategies.py`.
*   **Storage Management:** Saves processed documents and chunks to the designated storage location (`DOCUMENT_STORAGE_PATH`) and potentially stores metadata in the database.
*   **Task Dispatching:** Triggers downstream processing tasks for other services (e.g., embedding generation) by placing new tasks onto the Celery queue.

## Architecture & Technology

*   **Framework:** Celery
*   **Message Broker:** Redis (used by Celery)
*   **Language:** Python

## Interfaces & Communication

The Corpus Service does **not** expose any direct HTTP API endpoints. It communicates with other components exclusively through asynchronous mechanisms and shared resources:

1.  **Incoming Communication (Task Queue):**
    *   **Mechanism:** Celery tasks consumed from a Redis queue.
    *   **Producer:** Typically, the `api_gateway` service places tasks onto the queue after a user uploads a document via an endpoint like `POST /api/v1/documents`.
    *   **Example Task (Conceptual):** `corpus_service.tasks.process_document(document_id=<uuid>, source_filepath=<path>)`

2.  **Outgoing Communication (Task Queue):**
    *   **Mechanism:** Placing new Celery tasks onto the Redis queue for other services.
    *   **Consumers:** Potentially `embedding_service`, `graph_service`, or other future processing services.
    *   **Example Task (Conceptual):** `embedding_service.tasks.generate_embeddings_for_chunks(document_id=<uuid>, chunk_ids=[<uuid>, ...])`

3.  **Shared Resources:**
    *   **Database (PostgreSQL):** Reads document metadata and updates processing status (e.g., `PENDING`, `CHUNKING`, `COMPLETED`, `FAILED`).
    *   **Document Storage (Shared Volume):** Reads raw uploaded files from and writes processed chunks to a shared filesystem volume mapped to `DOCUMENT_STORAGE_PATH`. Access to this volume must be configured consistently across services that need it (e.g., `api_gateway`, `corpus_service`, potentially others).
    *   **Configuration (`.env`):** Relies on environment variables for database connection details, Redis connection details, storage paths, etc.

## Core Components (Internal)

*   `worker.py`: Defines the Celery application instance.
*   `tasks/`: Contains the definitions of the Celery tasks that the worker can execute (e.g., `process_document`).
*   `chunking_strategies.py`: Implements different algorithms for splitting documents into chunks.

## Running Locally (Development)

To run the Corpus Service worker locally (assuming dependencies are installed and Redis/Postgres are running):

1.  Ensure all necessary environment variables are set (see `.env.example`).
2.  Navigate to the `rag-backend` directory (the parent directory of `corpus_service`).
3.  Run the Celery worker:
    ```bash
    celery -A corpus_service.worker worker --loglevel=info
    ```

## Configuration

Key environment variables required:

*   `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
*   `REDIS_HOST`, `REDIS_PORT` (or `CELERY_BROKER_URL`)
*   `DOCUMENT_STORAGE_PATH`

Refer to `.env.example` in the project root for a complete list.