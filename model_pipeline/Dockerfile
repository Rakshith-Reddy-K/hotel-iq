# Production Dockerfile for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    PYTHONPATH=/app/backend 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Backend Code
COPY backend /app/backend

# Copy the Frontend Code (for /chat and /admin)
COPY frontend /app/frontend

# Create necessary directories
RUN mkdir -p /app/data/processed /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# We stay in /app/backend so main.py imports work relative to it
WORKDIR /app/backend 
# Run application (JSON format - recommended)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]