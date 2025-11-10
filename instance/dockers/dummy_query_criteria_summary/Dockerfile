FROM python:3.11-slim

# Install curl for healthcheck and uv for dependency management
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy application files
COPY pyproject.toml ./
COPY main.py ./
COPY start.sh ./
COPY data/ ./data/

# Make start script executable
RUN chmod +x start.sh

# Install dependencies using uv
RUN uv sync

# Expose port
EXPOSE 8000

# Set entry point
CMD ["./start.sh"]
