# Use an official Python runtime as a parent image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:/app/.venv/bin:$PATH"

# Install poetry
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get remove -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy dependency files from the root directory
COPY ../pyproject.toml ../poetry.lock ./

# Install project dependencies
# Use --no-root to avoid installing the project itself, only dependencies
# You might exclude dev dependencies here if they are not needed for the worker
# RUN poetry install --no-root --only main
# However, keeping them consistent with api_gateway might be simpler for now
RUN poetry install --no-root

# Copy the application code
# Copy the corpus_service code
COPY . /app/corpus_service
# Copy the common module (adjust path if it's located elsewhere)
COPY ../common /app/common

# Set the PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Define the command to run the Celery worker
# Use the same command as in docker-compose.yml
CMD ["celery", "-A", "corpus_service.worker", "worker", "--loglevel=info"]