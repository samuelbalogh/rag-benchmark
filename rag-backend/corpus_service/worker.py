"""Celery worker configuration for corpus service."""

from celery import Celery

# Initialize Celery app
app = Celery('corpus_service',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1',
             include=['corpus_service.tasks.processing',
                      'corpus_service.tasks.chunking'])

# Configure Celery
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

if __name__ == '__main__':
    app.start() 