FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy application files
COPY pyproject.toml uv.lock ./
COPY main.py ./
COPY start.sh ./

# Make start script executable
RUN chmod +x start.sh

# Install dependencies using uv
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Set entry point
CMD ["./start.sh"]
