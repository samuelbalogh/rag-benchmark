import os
from celery import Celery
import json
import uuid
from datetime import datetime

from common.logging import get_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Redis settings from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
BROKER_URL = os.getenv("CELERY_BROKER_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", f"redis://{REDIS_HOST}:{REDIS_PORT}/1")

# Create logger
logger = get_logger(__name__)

# Create Celery app
app = Celery(
    "corpus_service",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["corpus_service.tasks.chunking", "corpus_service.tasks.processing"],
)

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour time limit for tasks
    worker_prefetch_multiplier=1,  # Fetch one task at a time
    task_acks_late=True,  # Acknowledge tasks after execution
)

# Add periodic tasks
app.conf.beat_schedule = {
    "cleanup-old-processing-tasks": {
        "task": "corpus_service.tasks.processing.cleanup_old_tasks",
        "schedule": 3600.0,  # Every hour
    },
}

if __name__ == "__main__":
    app.start() 