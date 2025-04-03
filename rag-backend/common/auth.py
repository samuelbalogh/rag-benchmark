import os
from typing import Optional

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from common.database import get_db
from common.errors import AuthenticationError
from common.logging import get_logger
from common.models import ApiKey

# Get API key header name from environment
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")

# Create API key header scheme
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

# Logger for auth-related operations
logger = get_logger(__name__)


async def get_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> ApiKey:
    """
    Get and validate the API key from the request headers.
    
    Args:
        request: FastAPI request object
        api_key: API key from header
        db: Database session
        
    Returns:
        ApiKey: The validated API key
        
    Raises:
        AuthenticationError: If API key is missing or invalid
    """
    # Check if API key is provided
    if api_key is None:
        logger.warning("Missing API key in request", extra={"path": request.url.path})
        raise AuthenticationError("API key is required")
    
    # Query database for API key
    db_api_key = db.query(ApiKey).filter(ApiKey.key == api_key, ApiKey.enabled.is_(True)).first()
    
    # Check if API key is valid
    if db_api_key is None:
        logger.warning("Invalid API key", extra={"path": request.url.path})
        raise AuthenticationError("Invalid API key")
    
    logger.info("API key validated", extra={"api_key_id": str(db_api_key.id)})
    return db_api_key 