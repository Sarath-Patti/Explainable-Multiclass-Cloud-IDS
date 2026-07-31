# Official lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Environment variables to optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend source code and trained model artifacts
COPY backend /app/backend
COPY models /app/models

# Set working directory to backend application directory
WORKDIR /app/backend

# Expose FastAPI application port
EXPOSE 8000

# Start FastAPI application server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
