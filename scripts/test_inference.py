#!/usr/bin/env python3
"""
Test inference: query text → Query Tower → Qdrant ANN search → top-K places.
Demonstrates the full recommendation pipeline end-to-end.

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --query "วัดสวยๆ เชียงใหม่"
    python scripts/test_inference.py --query "คาเฟ่ริมทะเล ภูเก็ต"
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_DIR, TEXT_EMBED_DIM, EMBEDDING_DIM, HIDDEN_DIM,
    GNN_EMBED_DIM, NUM_CATEGORIES, NUM_PROVINCES, NUM_REGIONS,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Sample test queries ──
SAMPLE_QUERIES = [
    "วัดสวยๆ เชียงใหม่ ถ่ายรูป",
    "คาเฟ่น่ารัก กรุงเทพ",
    "ทะเลสวย ภูเก็ต ดำน้ำ",
    "น้ำตก อุทยานแห่งชาติ เขาใหญ่",
    "ตลาดนัด อาหารอร่อย เยาวราช",
    "temple beautiful chiang mai",
    "beach sunset krabi",
    "street food bangkok night market",
]


def load_query_tower(device: torch.device):
    """Load Query Tower — try TorchScript first, fallback to state_dict."""
    # Try TorchScript (no class definition needed)
    ts_path = MODEL_DIR / "query_tower_traced.pt"
    if ts_path.exists():
        logger.info(f"Loading TorchScript Query Tower: {ts_path}")
        model = torch.jit.load(str(ts_path), map_location=device)
        model.eval()
        return model

    # Fallback: state_dict
    sd_path = MODEL_DIR / "query_tower.pt"
    if sd_path.exists():
        logger.info(f"Loading Query Tower state_dict: {sd_path}")
        from src.models.query_tower import QueryTower
        model = QueryTower(
            text_dim=TEXT_EMBED_DIM,
            engagement_dim=4,
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBEDDING_DIM,
            dropout=0.0,
        )
        model.load_state_dict(torch.load(sd_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        return model

    raise FileNotFoundError("No query tower model found in models/")


def search_qdrant(query_vector: list[float], top_k: int = 10):
    """Search Qdrant for nearest places."""
    from qdrant_client import QdrantClient

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    # qdrant-client v2.x uses query_points instead of search
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return response.points


def main():
    parser = argparse.ArgumentParser(description="Test Two-Tower recommendation inference")
    parser.add_argument("--query", type=str, default=None,
                        help="Custom query text (Thai or English)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of results to return")
    parser.add_argument("--run_samples", action="store_true", default=False,
                        help="Run all sample queries")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load models ──
    logger.info("Loading text encoder (bge-m3)...")
    t0 = time.time()
    text_model = SentenceTransformer("BAAI/bge-m3")
    logger.info(f"  bge-m3 loaded in {time.time() - t0:.1f}s")

    logger.info("Loading Query Tower...")
    query_tower = load_query_tower(device)

    # ── Determine queries ──
    queries = []
    if args.query:
        queries.append(args.query)
    if args.run_samples or not args.query:
        queries.extend(SAMPLE_QUERIES)

    # ── Run inference ──
    for query_text in queries:
        print("\n" + "=" * 70)
        print(f"  Query: {query_text}")
        print("=" * 70)

        # Step 1: Encode text with bge-m3 → 1024d
        t1 = time.time()
        text_emb = text_model.encode(query_text, normalize_embeddings=True)
        text_emb_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)

        # Step 2: Engagement features (zero for pure text search)
        engagement = torch.zeros(1, 4, dtype=torch.float32).to(device)

        # Step 3: Query Tower → 128d embedding
        with torch.no_grad():
            query_emb = query_tower(text_emb_tensor, engagement)

        query_vector = query_emb.cpu().numpy().flatten().tolist()
        encode_time = time.time() - t1

        # Step 4: Search Qdrant
        t2 = time.time()
        results = search_qdrant(query_vector, top_k=args.top_k)
        search_time = time.time() - t2

        # ── Display results ──
        print(f"  Encode: {encode_time * 1000:.0f}ms | Search: {search_time * 1000:.0f}ms")
        print(f"  Top-{args.top_k} results:")
        print("-" * 70)

        for rank, hit in enumerate(results, 1):
            p = hit.payload
            name = p.get("name", "N/A")
            region = p.get("region", "")
            province_id = p.get("province_id", "")
            category_id = p.get("category_id", "")
            rating = p.get("google_avg_rating", 0)
            reviews = p.get("google_review_count", 0)
            score = hit.score

            print(f"  {rank:2d}. [{score:.4f}] {name}")
            print(f"      place_id={p.get('place_id')}, cat={category_id}, "
                  f"prov={province_id}, region={region}, "
                  f"rating={rating:.1f} ({reviews} reviews)")

    print("\n" + "=" * 70)
    print("  Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
