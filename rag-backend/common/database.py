"""Database connection and session management."""

import os
from contextlib import contextmanager
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use environment variables if available, otherwise use defaults
# During testing, we'll mock this connection
is_test_environment = os.environ.get("TESTING", "false").lower() == "true"

if is_test_environment:
    # Use SQLite in-memory for testing
    SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
else:
    DB_USER = os.environ.get("POSTGRES_USER", "postgres")
    DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
    DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
    DB_NAME = os.environ.get("POSTGRES_DB", "rag_benchmark")
    
    # Format the connection string with URL encoding for the password
    SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine - during test, this will be SQLite in-memory
if is_test_environment:
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Avoid actual connections during import time
    engine = None

# Create session factory
if is_test_environment:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    # Create session factory without binding to engine yet
    SessionLocal = sessionmaker(autocommit=False, autoflush=False)

# Create base class for declarative models
Base = declarative_base()


def init_db(db_url=None):
    """Initialize database connection and create tables.
    
    Args:
        db_url: Database URL (optional, uses default if not provided)
        
    Returns:
        SQLAlchemy engine
    """
    global engine
    if db_url is None:
        db_url = SQLALCHEMY_DATABASE_URL
    
    # Create engine
    engine = create_engine(db_url)
    
    # Create connection to test it
    with engine.connect() as conn:
        pass
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Update session factory to use this engine
    SessionLocal.configure(bind=engine)
    
    return engine


def get_db():
    """Get database session.
    
    Returns:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session(db_url):
    """Get database session for a specific URL.
    
    Args:
        db_url: Database URL
        
    Returns:
        SQLAlchemy session
    """
    engine = create_engine(db_url)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_factory() 