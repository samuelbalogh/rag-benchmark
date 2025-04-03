import os
import time
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from common.errors import handle_app_errors
from common.logging import get_logger

# Import routers
from api_gateway.routers import (
    documents,
    query,
    config,
    admin,
)

# Get environment variables
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Create logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Benchmarking API",
    description="API for RAG Benchmarking and Comparison",
    version="0.1.0",
    debug=DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID and logging middleware
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next: Callable) -> Response:
    # Generate request ID
    request_id = request.headers.get("X-Request-ID", str(time.time()))
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_host": request.client.host if request.client else None,
        },
    )
    
    # Set request start time
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {response.status_code}",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time_ms": int(process_time * 1000),
            },
        )
        
        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    except Exception as e:
        # Log error
        logger.error(
            f"Error processing request: {str(e)}",
            exc_info=True,
            extra={
                "request_id": request_id,
                "error": str(e),
            },
        )
        
        # Convert error to HTTP exception
        http_exception = handle_app_errors(e)
        
        # Create response
        response = Response(
            content=http_exception.detail,
            status_code=http_exception.status_code,
            media_type="application/json",
        )
        
        # Add headers to response
        response.headers["X-Request-ID"] = request_id
        
        return response

# Include routers
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(config.router, prefix="/api/v1", tags=["Configuration"])
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": os.getenv("API_KEY_HEADER", "X-API-Key"),
        }
    }
    
    # Add security requirement to all operations
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            operation["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=DEBUG) 