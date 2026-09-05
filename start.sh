#!/bin/bash
set -e

echo "=== Starting TechDocSystem Deployment ==="

# 1. Start FastAPI backend on internal port 8000
echo "Starting FastAPI backend on 127.0.0.1:8000..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# 2. Wait for FastAPI backend to be ready
echo "Waiting for FastAPI backend to become ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "FastAPI backend is live and healthy!"
        break
    fi
    sleep 1
done

# 3. Start Streamlit frontend on port 7860 (publicly exposed by Hugging Face)
echo "Starting Streamlit frontend on 0.0.0.0:7860..."
exec streamlit run frontend/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
