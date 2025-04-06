"""Celery worker configuration for corpus service."""

import os
from celery import Celery

# Get broker and backend URLs from environment, with defaults for local development
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://redis:6379/0')
result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/1')

# Create Celery instance
app = Celery('corpus_service')

# Configure Celery
app.conf.update(
    broker_url=broker_url,
    result_backend=result_backend,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Load task modules
app.autodiscover_tasks(['corpus_service.tasks.processing', 'corpus_service.tasks.chunking'])

# Print configuration for debugging
print(f"Final broker URL: {app.conf.broker_url}")
print(f"Final result backend: {app.conf.result_backend}")

if __name__ == '__main__':
    app.start() 