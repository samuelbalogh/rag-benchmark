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

ENV PATH="$POETRY_HOME/bin:$PATH"

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
# Exclude dev dependencies for a smaller image, but include them if needed for --reload
RUN poetry install --no-root

# Copy the application code
# Copy the api_gateway code
COPY . /app/api_gateway
# Copy the common module (adjust path if it's located elsewhere)
COPY ../common /app/common

# Expose the port the app runs on
EXPOSE 8003

# Define the command to run the application
# Use the same command as in docker-compose.yml, including --reload for development consistency
CMD ["uvicorn", "api_gateway.main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]