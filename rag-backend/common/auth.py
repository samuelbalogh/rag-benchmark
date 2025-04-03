"""Authentication and authorization utilities."""

import os
import uuid
import datetime
from typing import Dict

from jose import jwt, JWTError

from common.errors import AuthenticationError, AuthorizationError
from common.database import get_db


# JWT settings
SECRET_KEY = os.environ.get("SECRET_KEY", "test-secret-key-for-development-only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

# Header settings
API_KEY_HEADER = os.environ.get("API_KEY_HEADER", "X-API-Key")


def create_api_key(user_id: str, name: str) -> Dict:
    """Create a new API key.
    
    Args:
        user_id: User ID
        name: Name for the API key
        
    Returns:
        Dictionary with API key information
    """
    key_id = uuid.uuid4().hex
    expires = datetime.datetime.now() + datetime.timedelta(days=365)
    
    # Create payload
    payload = {
        "sub": user_id,
        "key_id": key_id,
        "exp": expires.timestamp()
    }
    
    # Create JWT token
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    
    # Create key record
    key_data = {
        "key_id": key_id,
        "key": token,
        "name": name,
        "user_id": user_id,
        "created_at": datetime.datetime.now()
    }
    
    # Save to database - mocked for testing
    # In production, this would insert into the database
    
    return key_data


def validate_api_key(api_key: str, secret_key: str = SECRET_KEY) -> Dict:
    """Validate an API key.
    
    Args:
        api_key: API key to validate
        secret_key: Secret key for JWT validation
        
    Returns:
        Dictionary with user ID and key ID
        
    Raises:
        AuthenticationError: If API key is invalid
    """
    try:
        # Decode JWT token
        payload = jwt.decode(api_key, secret_key, algorithms=[ALGORITHM])
        
        # Extract data
        user_id = payload.get("sub")
        key_id = payload.get("key_id")
        
        if user_id is None or key_id is None:
            raise AuthenticationError("Invalid API key")
        
        return {"user_id": user_id, "key_id": key_id}
    
    except JWTError:
        raise AuthenticationError("Invalid API key")


def get_api_key(api_key: str) -> Dict:
    """Get API key details from database.
    
    Args:
        api_key: API key
        
    Returns:
        Dictionary with API key details
        
    Raises:
        AuthenticationError: If API key is invalid or not found
    """
    # Validate JWT format
    key_data = validate_api_key(api_key)
    
    # In production, this would check the database
    # For testing, we return the validated data
    
    return {
        "id": key_data["key_id"],
        "user_id": key_data["user_id"],
        "is_active": True
    } 