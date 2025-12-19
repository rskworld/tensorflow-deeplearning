# TensorFlow Deep Learning Project Dockerfile
# Author: RSK World
# Website: https://rskworld.in
# Email: help@rskworld.in
# Phone: +91 93305 39277

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY api/requirements.txt api_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models data logs checkpoints

# Expose API port
EXPOSE 5000

# Default command (can be overridden)
CMD ["python", "api/server.py"]
