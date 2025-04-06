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
    PYSETUP_PATH="/opt/pysetup"

ENV PATH="/app/.venv/bin:$POETRY_HOME/bin:$PATH"


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

# Install project dependencies with verbose output
RUN poetry install --no-root -vvv

# Download necessary NLTK data during build
RUN python -m nltk.downloader punkt stopwords

# Copy the application code
# Create the corpus_service directory structure correctly for imports
COPY . /app
# Copy the common module (adjust path if it's located elsewhere)
COPY ../common /app/common

# Set the PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Define the command to run the Celery worker
# Use the same command as in docker-compose.yml
CMD ["poetry", "run", "python", "-m", "celery", "-A", "corpus_service.worker", "worker", "--loglevel=info"]