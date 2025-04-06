from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException, status


class AppError(Exception):
    """Base exception class for application errors."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "internal_error",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }
        
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict(),
        )


class ResourceNotFoundError(AppError):
    """Raised when a requested resource is not found."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="resource_not_found",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )


class ValidationError(AppError):
    """Raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="validation_error",
            details={"field_errors": field_errors or {}},
        )


class AuthenticationError(AppError):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="authentication_error",
        )


class AuthorizationError(AppError):
    """Raised when user is not authorized to access a resource."""
    
    def __init__(
        self,
        message: str = "Not authorized to access this resource",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="authorization_error",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
            },
        )


class ServiceError(AppError):
    """Raised when an external service fails."""
    
    def __init__(
        self,
        message: str,
        service_name: str,
        service_error: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="service_error",
            details={
                "service_name": service_name,
                "service_error": service_error,
            },
        )


class DuplicateResourceError(AppError):
    """Raised when a resource already exists."""
    
    def __init__(
        self,
        message: str,
        resource_type: str,
        identifier: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="duplicate_resource",
            details={
                "resource_type": resource_type,
                "identifier": identifier or {},
            },
        )


class TaskError(AppError):
    """Raised when an async task fails."""
    
    def __init__(
        self,
        message: str,
        task_id: str,
        task_type: str,
        task_error: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="task_error",
            details={
                "task_id": task_id,
                "task_type": task_type,
                "task_error": task_error,
            },
        )


class EvaluationError(AppError):
    """Raised when evaluation metrics calculation fails."""
    
    def __init__(
        self,
        message: str = "Evaluation metrics calculation failed",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="evaluation_error",
            details=details or {},
        )


class RAGError(AppError):
    """Raised when a RAG operation fails."""
    
    def __init__(
        self,
        message: str = "RAG operation failed",
        code: str = "rag_error",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code=code,
            details=details or {},
        )


def handle_app_errors(error: Exception) -> HTTPException:
    """
    Convert application errors to FastAPI HTTP exceptions.
    
    Args:
        error: The exception to convert
        
    Returns:
        HTTPException
    """
    if isinstance(error, AppError):
        return error.to_http_exception()
        
    # For unexpected errors, return a generic 500 error
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error_code": "internal_error",
            "message": "An unexpected error occurred",
            "details": {"error": str(error)},
        },
    ) 