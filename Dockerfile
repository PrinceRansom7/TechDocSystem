FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    API_BASE_URL=http://127.0.0.1:8000 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a non-root user (Hugging Face Spaces default UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . .

# Ensure data directories exist and have proper permissions
RUN mkdir -p data/chroma_db data/uploads

# Ensure startup script is executable
RUN chmod +x start.sh

# Expose Hugging Face Space default port
EXPOSE 7860

# Run the startup script
CMD ["./start.sh"]
