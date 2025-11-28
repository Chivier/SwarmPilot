FROM python:3.11-slim

# Install runtime dependencies
# - curl: for healthcheck
# - libgl1-mesa-glx, libglib2.0-0: OpenCV dependencies (required by EasyOCR)
# - libgomp1: OpenMP for parallel processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy application files
COPY pyproject.toml ./
COPY main.py ./
COPY start.sh ./

# Make start script executable
RUN chmod +x start.sh

# Install dependencies using uv
RUN uv sync

# Expose port
EXPOSE 8000

# Health check (longer start period for model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entry point
CMD ["./start.sh"]
