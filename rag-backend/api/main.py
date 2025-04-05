"""Main API module for the RAG benchmark platform."""

import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.routes import query
from common.logging import get_logger
from common.config import get_settings

# Initialize logger
logger = get_logger(__name__)

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="RAG Benchmark API",
    description="API for the Retrieval-Augmented Generation Benchmark platform",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router)


@app.get("/health", tags=["health"])
async def health_check():
    """Check API health."""
    return {"status": "healthy", "version": app.version}


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the RAG Benchmark API",
        "docs_url": "/docs",
        "version": app.version
    }


def start_api():
    """Start the API server."""
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("API_PORT", 8000))
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level="info"
    )


if __name__ == "__main__":
    start_api() 