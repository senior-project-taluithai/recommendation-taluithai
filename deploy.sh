#!/usr/bin/env bash
# Deploy recommendation service to Google Cloud Run
# Usage: ./deploy.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-taluithai}"
REGION="${GCP_REGION:-asia-southeast1}"
SERVICE_NAME="recommendation-service"
IMAGE_NAME="asia-southeast1-docker.pkg.dev/${PROJECT_ID}/taluithai/${SERVICE_NAME}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Deploying ${SERVICE_NAME} to Cloud Run                    ║"
echo "║  Project: ${PROJECT_ID}                                    ║"
echo "║  Region:  ${REGION}                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Step 1: Build & Push image ────────────────────────────────────────────────
echo ""
echo "▶ Step 1: Building container image..."

# Create Artifact Registry repo if not exists
gcloud artifacts repositories create taluithai \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || true

# Configure Docker auth for Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build and push (using Cloud Build for faster builds)
echo "  Building with Cloud Build..."
gcloud builds submit \
    --tag "${IMAGE_NAME}:latest" \
    --project "${PROJECT_ID}" \
    --timeout=600s

# ── Step 2: Deploy to Cloud Run ──────────────────────────────────────────────
echo ""
echo "▶ Step 2: Deploying to Cloud Run..."

gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}:latest" \
    --platform managed \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --port 8080 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --timeout 60s \
    --concurrency 80 \
    --set-env-vars "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" \
    --set-env-vars "QDRANT_HOST=${QDRANT_HOST:-db-taluithai.oswinfalk.xyz}" \
    --set-env-vars "QDRANT_PORT=${QDRANT_PORT:-6333}" \
    --set-env-vars "QDRANT_COLLECTION=${QDRANT_COLLECTION:-place_recommendations}" \
    --allow-unauthenticated \
    --quiet

# ── Step 3: Verify ───────────────────────────────────────────────────────────
echo ""
echo "▶ Step 3: Verifying deployment..."

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --platform managed \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format "value(status.url)")

echo "  Service URL: ${SERVICE_URL}"

# Health check
echo "  Testing /health..."
curl -s "${SERVICE_URL}/health" | python3 -m json.tool

# Quick recommendation test
echo ""
echo "  Testing /recommend..."
curl -s "${SERVICE_URL}/recommend?query=%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B8%AA%E0%B8%A7%E0%B8%A2%E0%B9%86&top_k=3" | python3 -m json.tool

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Deployment complete!                                    ║"
echo "║  URL: ${SERVICE_URL}                                       ║"
echo "║  Docs: ${SERVICE_URL}/docs                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
