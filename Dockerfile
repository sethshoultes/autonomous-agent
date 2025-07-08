# ============================================================================
# Multi-stage Dockerfile for Autonomous Agent System
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base Image with Security Hardening
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bullseye as base

# Set environment variables for security and optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Poetry
RUN pip install --no-cache-dir poetry==$POETRY_VERSION

# Create non-root user for security
RUN groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent

# Create app directory with proper permissions
RUN mkdir -p /app /app/logs /app/data \
    && chown -R agent:agent /app

WORKDIR /app

# -----------------------------------------------------------------------------
# Stage 2: Dependencies Installation
# -----------------------------------------------------------------------------
FROM base as deps

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --only=main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# -----------------------------------------------------------------------------
# Stage 3: Development Environment
# -----------------------------------------------------------------------------
FROM deps as development

# Install development dependencies
RUN poetry install --no-root

# Copy application code
COPY --chown=agent:agent . .

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development
CMD ["poetry", "run", "python", "src/main.py"]

# -----------------------------------------------------------------------------
# Stage 4: Production Environment
# -----------------------------------------------------------------------------
FROM deps as production

# Copy only necessary application files
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent pyproject.toml ./

# Install the application
RUN poetry install --only=main --no-dev

# Create production directories
RUN mkdir -p /app/logs /app/data /app/config \
    && chown -R agent:agent /app

# Switch to non-root user
USER agent

# Set production environment with security
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app \
    JWT_SECRET_FILE=/run/secrets/jwt_secret \
    SECURE_HEADERS=true

# Expose port
EXPOSE 8000

# Health check with longer intervals for production
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with proper signal handling
CMD ["poetry", "run", "python", "-m", "src.main"]

# -----------------------------------------------------------------------------
# Stage 5: Security Scanning (Optional)
# -----------------------------------------------------------------------------
FROM production as security-scan

# Switch back to root for security scanning
USER root

# Install security scanning tools
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3-pip \
    && pip install --no-cache-dir \
    bandit \
    safety \
    && rm -rf /var/lib/apt/lists/*

# Run security scans
RUN bandit -r src/ -f json -o /tmp/bandit-report.json || true
RUN safety check --json --output /tmp/safety-report.json || true

# Switch back to non-root user
USER agent

# -----------------------------------------------------------------------------
# Stage 6: Minimal Production Image
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bullseye as minimal

# Copy only runtime essentials
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    curl \
    libpq5 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r agent && useradd -r -g agent -d /app -s /bin/bash agent

# Create app directory
RUN mkdir -p /app /app/logs /app/data \
    && chown -R agent:agent /app

WORKDIR /app

# Copy virtual environment from deps stage
COPY --from=deps --chown=agent:agent /app/.venv /app/.venv

# Copy application code
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent pyproject.toml ./

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["python", "-m", "src.main"]

# -----------------------------------------------------------------------------
# Labels for container metadata
# -----------------------------------------------------------------------------
LABEL maintainer="Seth Shoultes <seth@example.com>" \
    version="0.1.0" \
    description="Autonomous Agent System - Production Ready Container" \
    org.opencontainers.image.title="Autonomous Agent System" \
    org.opencontainers.image.description="A privacy-focused autonomous agent system" \
    org.opencontainers.image.version="0.1.0" \
    org.opencontainers.image.created="2024-01-01T00:00:00Z" \
    org.opencontainers.image.source="https://github.com/username/autonomous-agent" \
    org.opencontainers.image.licenses="MIT"