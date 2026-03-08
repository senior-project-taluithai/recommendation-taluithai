"""
Recommendation serving service — FastAPI app for Cloud Run.

Architecture:
  query text → OpenRouter bge-m3 (1024d) → Query Tower (128d) → Qdrant ANN → top-K places

Env vars:
  OPENROUTER_API_KEY  — required
  QDRANT_HOST         — default: db-taluithai.oswinfalk.xyz
  QDRANT_PORT         — default: 6333
  QDRANT_COLLECTION   — default: place_recommendations
  MODEL_PATH          — default: models/query_tower_traced.pt
  PORT                — default: 8080 (Cloud Run injects this)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "baai/bge-m3")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "db-taluithai.oswinfalk.xyz")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "place_recommendations")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/query_tower_traced.pt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serving")

# ── Global state (loaded on startup) ─────────────────────────────────────────
query_tower: torch.jit.ScriptModule | None = None
qdrant: QdrantClient | None = None
http_client: httpx.AsyncClient | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model & connections on startup, close on shutdown."""
    global query_tower, qdrant, http_client

    t0 = time.time()

    # 1. Load Query Tower (TorchScript, ~2.7MB)
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")
    query_tower = torch.jit.load(str(model_path), map_location="cpu")
    query_tower.eval()
    logger.info(f"Query Tower loaded from {model_path} ({model_path.stat().st_size / 1e6:.1f}MB)")

    # 2. Connect to Qdrant
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    info = qdrant.get_collection(QDRANT_COLLECTION)
    logger.info(f"Qdrant connected: {QDRANT_HOST}:{QDRANT_PORT}, collection={QDRANT_COLLECTION} ({info.points_count} points)")

    # 3. HTTP client for OpenRouter
    http_client = httpx.AsyncClient(timeout=30.0)

    logger.info(f"Startup complete in {time.time() - t0:.2f}s")

    yield  # ── App runs ──

    # Shutdown
    await http_client.aclose()
    logger.info("Shutdown complete")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Taluithai Recommendation Service",
    description="Two-Tower recommendation: query → bge-m3 → Query Tower → Qdrant ANN search",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PlaceResult(BaseModel):
    place_id: int
    name: str
    score: float
    category_id: int | None = None
    province_id: int | None = None
    region: str | None = None
    google_avg_rating: float | None = None
    google_review_count: int | None = None
    latitude: float | None = None
    longitude: float | None = None


class RecommendResponse(BaseModel):
    query: str
    results: list[PlaceResult]
    timing: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    qdrant_connected: bool
    collection: str
    points_count: int


# ── Embedding helper ──────────────────────────────────────────────────────────
async def encode_text_openrouter(text: str) -> list[float]:
    """Call OpenRouter bge-m3 API → 1024d normalized embedding."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set")

    resp = await http_client.post(
        OPENROUTER_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter embedding failed ({resp.status_code}): {resp.text[:200]}",
        )

    data = resp.json()
    embedding = data["data"][0]["embedding"]

    # Normalize (OpenRouter may or may not return normalized vectors)
    arr = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm

    return arr.tolist()


# ── Reranking config ──────────────────────────────────────────────────────────
CATEGORY_BOOST = float(os.environ.get("CATEGORY_BOOST", "0.15"))   # score boost for preferred category
REGION_BOOST = float(os.environ.get("REGION_BOOST", "0.10"))       # score boost for preferred region
RERANK_OVERSAMPLE = int(os.environ.get("RERANK_OVERSAMPLE", "3"))  # fetch N× candidates for reranking


def rerank_results(
    results: list[PlaceResult],
    preferred_category_ids: list[int],
    preferred_regions: list[str],
    top_k: int,
) -> list[PlaceResult]:
    """
    Post-retrieval reranking: boost score for matching category/region preferences.
    
    - Category match: +CATEGORY_BOOST
    - Region match: +REGION_BOOST
    - Re-sort by boosted score, return top_k
    """
    if not preferred_category_ids and not preferred_regions:
        return results[:top_k]

    cat_set = set(preferred_category_ids)
    region_set = set(preferred_regions)

    boosted = []
    for r in results:
        boost = 0.0
        if r.category_id and r.category_id in cat_set:
            boost += CATEGORY_BOOST
        if r.region and r.region in region_set:
            boost += REGION_BOOST
        boosted.append(PlaceResult(
            place_id=r.place_id,
            name=r.name,
            score=round(r.score + boost, 4),
            category_id=r.category_id,
            province_id=r.province_id,
            region=r.region,
            google_avg_rating=r.google_avg_rating,
            google_review_count=r.google_review_count,
            latitude=r.latitude,
            longitude=r.longitude,
        ))

    boosted.sort(key=lambda x: x.score, reverse=True)
    return boosted[:top_k]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — also useful for Cloud Run startup probe."""
    points_count = 0
    qdrant_ok = False
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        points_count = info.points_count
        qdrant_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if query_tower and qdrant_ok else "degraded",
        model_loaded=query_tower is not None,
        qdrant_connected=qdrant_ok,
        collection=QDRANT_COLLECTION,
        points_count=points_count,
    )


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    query: str = Query(..., min_length=1, max_length=500, description="Search query (Thai or English)"),
    top_k: int = Query(10, ge=1, le=100, description="Number of results"),
    engagement_plays: float = Query(0.0, description="Log-normalized play count"),
    engagement_likes: float = Query(0.0, description="Log-normalized like count"),
    engagement_shares: float = Query(0.0, description="Log-normalized share count"),
    engagement_collects: float = Query(0.0, description="Log-normalized collect count"),
    preferred_category_ids: str = Query("", description="Comma-separated preferred category IDs (e.g. '3,8')"),
    preferred_regions: str = Query("", description="Comma-separated preferred regions (e.g. 'North,South')"),
):
    """
    Recommend places based on text query with optional preference-based reranking.
    
    Pipeline: query → OpenRouter bge-m3 (1024d) → Query Tower (128d) → Qdrant ANN → rerank → top-K
    """
    timings = {}

    # Parse preference params
    pref_cats = [int(x) for x in preferred_category_ids.split(",") if x.strip().isdigit()]
    pref_regions = [x.strip() for x in preferred_regions.split(",") if x.strip()]
    has_prefs = bool(pref_cats or pref_regions)

    # If reranking, fetch more candidates for better recall
    fetch_k = top_k * RERANK_OVERSAMPLE if has_prefs else top_k

    # Step 1: Encode text via OpenRouter bge-m3 → 1024d
    t1 = time.time()
    text_embedding = await encode_text_openrouter(query)
    timings["embed_ms"] = round((time.time() - t1) * 1000, 1)

    # Step 2: Query Tower → 128d
    t2 = time.time()
    text_tensor = torch.tensor([text_embedding], dtype=torch.float32)
    engagement_tensor = torch.tensor(
        [[engagement_plays, engagement_likes, engagement_shares, engagement_collects]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        query_emb = query_tower(text_tensor, engagement_tensor)

    query_vector = query_emb.numpy().flatten().tolist()
    timings["tower_ms"] = round((time.time() - t2) * 1000, 1)

    # Step 3: Qdrant ANN search
    t3 = time.time()
    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
    )
    timings["search_ms"] = round((time.time() - t3) * 1000, 1)

    # Step 4: Format results
    results = []
    for hit in response.points:
        p = hit.payload
        results.append(PlaceResult(
            place_id=p.get("place_id", 0),
            name=p.get("name", "N/A"),
            score=round(hit.score, 4),
            category_id=p.get("category_id"),
            province_id=p.get("province_id"),
            region=p.get("region"),
            google_avg_rating=p.get("google_avg_rating"),
            google_review_count=p.get("google_review_count"),
            latitude=p.get("latitude"),
            longitude=p.get("longitude"),
        ))

    # Step 5: Rerank by user preferences (if provided)
    t4 = time.time()
    if has_prefs:
        results = rerank_results(results, pref_cats, pref_regions, top_k)
    else:
        results = results[:top_k]
    timings["rerank_ms"] = round((time.time() - t4) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)

    return RecommendResponse(
        query=query,
        results=results,
        timing=timings,
    )


@app.get("/recommend/batch", response_model=list[RecommendResponse])
async def recommend_batch(
    queries: str = Query(..., description="Comma-separated queries"),
    top_k: int = Query(10, ge=1, le=100),
    preferred_category_ids: str = Query("", description="Comma-separated preferred category IDs"),
    preferred_regions: str = Query("", description="Comma-separated preferred regions"),
):
    """Batch recommend for multiple queries with optional preference reranking."""
    query_list = [q.strip() for q in queries.split(",") if q.strip()]
    if len(query_list) > 10:
        raise HTTPException(status_code=400, detail="Max 10 queries per batch")

    results = []
    for q in query_list:
        r = await recommend(
            query=q,
            top_k=top_k,
            preferred_category_ids=preferred_category_ids,
            preferred_regions=preferred_regions,
        )
        results.append(r)

    return results


# ── Main (for local dev) ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("serving.app:app", host="0.0.0.0", port=port, reload=True)
