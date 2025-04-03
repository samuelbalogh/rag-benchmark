import json
import logging
import os
import sys
import uuid
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from dotenv import load_dotenv

load_dotenv()

# Type variable for preserving return type in decorators
F = TypeVar("F", bound=Callable[..., Any])

# Configure the logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json").lower()

# Create a custom formatter for structured logs
class JsonFormatter(logging.Formatter):
    """Format logs as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "path": record.pathname,
            "line": record.lineno,
        }
        
        # Add any additional attributes set on the record
        for key, value in record.__dict__.items():
            if key not in ("timestamp", "level", "message", "logger", "path", "line", 
                          "msg", "args", "exc_info", "exc_text", "levelname", 
                          "levelno", "pathname", "filename", "module", "name", 
                          "lineno", "funcName", "created", "msecs", "relativeCreated", 
                          "thread", "threadName", "processName", "process", "asctime"):
                log_data[key] = value
        
        # Include exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        
        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level based on environment variable
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create a handler for console output
    handler = logging.StreamHandler(sys.stdout)
    
    # Choose formatter based on environment variable
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        ))
    
    logger.addHandler(handler)
    
    return logger


def with_logging(logger: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator to add logging to functions.
    
    Args:
        logger: Optional logger to use. If None, a logger will be created.
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Create a logger if not provided
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
            
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate a unique ID for this function call
            correlation_id = str(uuid.uuid4())
            
            # Log function call
            logger.info(
                f"Function {func.__name__} called",
                extra={
                    "correlation_id": correlation_id,
                    "function": func.__name__,
                    "module": func.__module__,
                },
            )
            
            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"Function {func.__name__} completed",
                    extra={
                        "correlation_id": correlation_id,
                        "function": func.__name__,
                        "module": func.__module__,
                    },
                )
                return result
            except Exception as e:
                logger.error(
                    f"Function {func.__name__} failed: {str(e)}",
                    exc_info=True,
                    extra={
                        "correlation_id": correlation_id,
                        "function": func.__name__,
                        "module": func.__module__,
                        "error": str(e),
                    },
                )
                raise
                
        return cast(F, wrapper)
    
    return decorator 