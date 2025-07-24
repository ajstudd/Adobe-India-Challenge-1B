# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY pdf_parser.py .
COPY embedding_model.py .
COPY ranker.py .
COPY utils.py .

# Copy pre-downloaded model (this directory should exist before building)
COPY models/ ./models/

# Create required directories
RUN mkdir -p inputs/docs outputs

# Copy input files (if they exist)
COPY inputs/ ./inputs/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for future web interface)
EXPOSE 8000

# Health check to verify model is available
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from embedding_model import EmbeddingModel; em = EmbeddingModel(); print('Model OK')" || exit 1

# Run the main application
CMD ["python", "main.py"]
