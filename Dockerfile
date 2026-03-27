# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebook and supporting files
COPY app.py .
COPY run_analysis.py .

# Copy static files
COPY static/ static/
COPY templates/ templates/

# Create output directory
RUN mkdir -p /app/outputs

# Expose Flask port
EXPOSE 5000

# Default command: run Flask app
CMD ["python", "app.py"]
