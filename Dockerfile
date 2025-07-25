# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout 300 --retries 3 -r requirements.txt

# Copy application code
COPY main.py .
COPY pdf_parser.py .
COPY embedding_model.py .
COPY ranker.py .
COPY utils.py .
COPY download_model.py .

# Create models directory
RUN mkdir -p models

# Download model during build (with internet access)
RUN python download_model.py

# Create required directories
RUN mkdir -p input/PDFs output

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (we will use it in the next round)
EXPOSE 8000

# Health check to verify model is available
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from embedding_model import EmbeddingModel; em = EmbeddingModel(); print('Model OK')" || exit 1

CMD ["python", "main.py"]
