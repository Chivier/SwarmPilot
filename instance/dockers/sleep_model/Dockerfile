# ============ Build Stage ============
FROM python:3.11-slim AS builder

# Install uv (official standalone installer is much smaller than pip install)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only dependency file first (better layer caching)
COPY pyproject.toml ./

# Install dependencies to system (no venv overhead in container)
RUN uv sync --no-dev --no-install-project

# ============ Runtime Stage ============
FROM python:3.11-slim

# Install only runtime dependency (curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application files
COPY main.py start.sh ./

# Make start script executable
RUN chmod +x start.sh

# Use venv's Python
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["./start.sh"]