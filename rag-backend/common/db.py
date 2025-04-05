"""Database connection module for the RAG benchmark platform."""

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Generator, Any

from common.config import get_db_url
from common.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection.
    
    Yields:
        Database connection
    """
    conn = None
    try:
        # Connect to database
        conn = psycopg2.connect(
            get_db_url(),
            cursor_factory=RealDictCursor
        )
        yield conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise
    finally:
        if conn is not None:
            conn.close()


def execute_query(query: str, params: Any = None, fetch: bool = True) -> Any:
    """Execute a database query.
    
    Args:
        query: SQL query
        params: Query parameters
        fetch: Whether to fetch results
        
    Returns:
        Query results if fetch is True, otherwise None
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(query, params)
                
                if fetch:
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return None
            except Exception as e:
                conn.rollback()
                logger.error(f"Error executing query: {str(e)}")
                raise 