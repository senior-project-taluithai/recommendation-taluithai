# ── Stage 1: Build ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install only CPU PyTorch (much smaller than full CUDA version)
COPY serving/requirements.txt .
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code
COPY serving/ ./serving/

# Copy model file (only query_tower_traced.pt, 2.7MB)
COPY models/query_tower_traced.pt ./models/query_tower_traced.pt

# Set env defaults
ENV MODEL_PATH=models/query_tower_traced.pt \
    QDRANT_HOST=db-taluithai.oswinfalk.xyz \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION=place_recommendations \
    PORT=8080

EXPOSE 8080

# Cloud Run expects the app to listen on $PORT
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
