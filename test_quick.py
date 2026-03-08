#!/usr/bin/env python3
"""Quick test: Query Tower + Qdrant search on local machine."""
import torch
import numpy as np

# 1. Load Query Tower (TorchScript)
qt = torch.jit.load("models/query_tower_traced.pt", map_location="cpu")
qt.eval()
print("✓ Query Tower loaded")

# 2. Load bge-m3
from sentence_transformers import SentenceTransformer
print("  Loading bge-m3 (first time is slow ~30s)...")
tm = SentenceTransformer("BAAI/bge-m3")
print("✓ bge-m3 loaded")

# 3. Test queries
queries = [
    "วัดสวยๆ เชียงใหม่",
    "คาเฟ่น่ารัก กรุงเทพ",
    "ทะเลสวย ภูเก็ต",
    "น้ำตก อุทยานแห่งชาติ",
    "street food bangkok",
]

from qdrant_client import QdrantClient
from qdrant_client.models import models
from config import QDRANT_HOST, QDRANT_PORT
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
print("✓ Qdrant connected\n")

for query in queries:
    # Encode
    text_emb = tm.encode(query, normalize_embeddings=True, show_progress_bar=False)
    text_tensor = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0)
    engagement = torch.zeros(1, 4)
    
    with torch.no_grad():
        q_emb = qt(text_tensor, engagement)
    
    q_vec = q_emb.numpy().flatten().tolist()
    
    # Search using query_points (qdrant-client v2.x API)
    response = client.query_points(
        collection_name="place_recommendations",
        query=q_vec,
        limit=5,
        with_payload=True,
    )
    
    print("=" * 60)
    print(f"  Query: {query}")
    print("-" * 60)
    for i, hit in enumerate(response.points, 1):
        p = hit.payload
        name = p.get("name", "?")
        region = p.get("region", "")
        rating = p.get("google_avg_rating", 0)
        print(f"  {i}. [{hit.score:.4f}] {name}  (region={region}, rating={rating:.1f})")
    print()

print("✓ Test complete!")
