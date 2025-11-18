# HotelIQ Backend Dockerfile
# ==========================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Copy frontend from parent directory
COPY ../frontend /app/frontend

# Create directories for data (will be populated from GCS on startup)
RUN mkdir -p /app/data/processed /app/logs

# Copy booking requests from parent directory
COPY ../booking_requests.json /app/booking_requests.json

# Expose port
EXPOSE 8000

# Health check (disabled during startup to allow data download)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# Note: Data files will be downloaded from GCS on startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

