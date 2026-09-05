#!/bin/bash
set -e

# Cleanup child processes on script exit or interrupt
trap 'kill $(jobs -p) 2>/dev/null' SIGINT SIGTERM EXIT

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

# Render dynamically injects $PORT (default 10000 on Render web services)
PORT="${PORT:-10000}"

# 3. Start Streamlit frontend bound to 0.0.0.0:$PORT
echo "Starting Streamlit frontend on 0.0.0.0:${PORT}..."
exec streamlit run frontend/app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false \
    --server.headless=true
