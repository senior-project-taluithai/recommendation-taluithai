#!/usr/bin/env python3
"""
Step 4: Export trained Two-Tower item embeddings to Qdrant.

Pre-computes item embeddings for all 20K+ places and upserts
them into a Qdrant collection for real-time ANN retrieval.

Also exports the Query Tower model for inference use.

Usage:
    python scripts/04_export_qdrant.py
    python scripts/04_export_qdrant.py --model models/two_tower_best.pt
    python scripts/04_export_qdrant.py --dry_run   # Compute only, skip Qdrant upload
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_DIR, MODEL_DIR, EMBED_DIR,
    EMBEDDING_DIM, HIDDEN_DIM, TEXT_EMBED_DIM,
    GNN_EMBED_DIM, NUM_CATEGORIES, NUM_PROVINCES, NUM_REGIONS,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _region_to_id(region: str) -> int:
    mapping = {
        "North": 1, "Central": 2,
        "Northeast": 3, "West": 4,
        "East": 5, "South": 6,
    }
    return mapping.get(region, 0)


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained TwoTowerModel from checkpoint."""
    from src.models.two_tower import TwoTowerModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Use args from checkpoint or defaults
    saved_args = checkpoint.get("args", {})
    
    model = TwoTowerModel(
        text_dim=saved_args.get("text_dim", TEXT_EMBED_DIM),
        engagement_dim=4,
        hidden_dim=saved_args.get("hidden_dim", HIDDEN_DIM),
        embed_dim=saved_args.get("embed_dim", EMBEDDING_DIM),
        numerical_dim=5,
        graph_dim=saved_args.get("graph_dim", GNN_EMBED_DIM),
        cat_embed_dim=16,
        dropout=0.0,  # no dropout at inference
        num_categories=checkpoint.get("num_categories", NUM_CATEGORIES),
        num_provinces=checkpoint.get("num_provinces", NUM_PROVINCES),
        num_regions=checkpoint.get("num_regions", NUM_REGIONS),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    if "metrics" in checkpoint:
        logger.info(f"  Checkpoint metrics: {checkpoint['metrics']}")
    if "epoch" in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
    
    return model


def compute_item_embeddings(
    model, places_df, item_text_embeds, graph_embeds,
    cat_id_map, prov_id_map,
    device, batch_size=512,
) -> tuple[np.ndarray, list[int]]:
    """Pre-compute item embeddings for all places."""
    place_ids = places_df["place_id"].tolist()
    all_embs = []
    
    for i in tqdm(range(0, len(place_ids), batch_size), desc="Computing item embeddings"):
        batch_pids = place_ids[i:i + batch_size]
        
        # Text embeddings
        text_embs = torch.stack([
            torch.tensor(item_text_embeds.get(pid, np.zeros(TEXT_EMBED_DIM)), dtype=torch.float32)
            for pid in batch_pids
        ]).to(device)
        
        # Categorical features (use sequential ID maps)
        cats, nums = [], []
        for pid in batch_pids:
            row = places_df[places_df["place_id"] == pid].iloc[0]
            raw_cat = int(row["category_id"]) if pd.notna(row["category_id"]) else 0
            raw_prov = int(row["province_id"]) if pd.notna(row["province_id"]) else 0
            cats.append([
                cat_id_map.get(raw_cat, 0),
                prov_id_map.get(raw_prov, 0),
                _region_to_id(row.get("region", "")),
            ])
            nums.append([
                float(row["google_avg_rating"]) / 5.0,
                np.log1p(float(row["google_review_count"])) / 15.0,
                (float(row["latitude"]) - 13.0) / 7.0,
                (float(row["longitude"]) - 100.0) / 5.0,
                (len(row["wongnai_genres"]) if isinstance(row["wongnai_genres"], list) else 0) / 10.0,
            ])
        
        cat_tensor = torch.tensor(cats, dtype=torch.long).to(device)
        num_tensor = torch.tensor(nums, dtype=torch.float32).to(device)
        g_embs = torch.stack([
            torch.tensor(graph_embeds.get(pid, np.zeros(128)), dtype=torch.float32)
            for pid in batch_pids
        ]).to(device)
        
        with torch.no_grad():
            emb = model.encode_item(text_embs, cat_tensor, num_tensor, g_embs)
        all_embs.append(emb.cpu().numpy())
    
    return np.vstack(all_embs), place_ids


def upload_to_qdrant(
    place_ids: list[int],
    embeddings: np.ndarray,
    places_df: pd.DataFrame,
    collection_name: str,
    host: str,
    port: int,
    batch_size: int = 100,
):
    """Create Qdrant collection and upsert all place embeddings with metadata."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        OptimizersConfigDiff, HnswConfigDiff,
    )
    
    client = QdrantClient(host=host, port=port, timeout=120)
    
    # Recreate collection
    dim = embeddings.shape[1]
    
    if client.collection_exists(collection_name):
        logger.info(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)
    
    logger.info(f"Creating collection: {collection_name} (dim={dim}, cosine)")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=dim,
            distance=Distance.COSINE,
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=10000,
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
    )
    
    # Build place_id → row lookup for payload
    place_lookup = places_df.set_index("place_id")
    
    # Upsert in batches
    total = len(place_ids)
    for i in tqdm(range(0, total, batch_size), desc="Uploading to Qdrant"):
        batch_ids = place_ids[i:i + batch_size]
        batch_embs = embeddings[i:i + batch_size]
        
        points = []
        for pid, emb in zip(batch_ids, batch_embs):
            payload = {"place_id": int(pid)}
            
            if int(pid) in place_lookup.index:
                row = place_lookup.loc[int(pid)]
                payload.update({
                    "name": str(row.get("name", "")),
                    "category_id": int(row["category_id"]) if pd.notna(row.get("category_id")) else None,
                    "province_id": int(row["province_id"]) if pd.notna(row.get("province_id")) else None,
                    "region": str(row.get("region", "")),
                    "google_avg_rating": float(row.get("google_avg_rating", 0)),
                    "google_review_count": int(row.get("google_review_count", 0)),
                    "latitude": float(row.get("latitude", 0)),
                    "longitude": float(row.get("longitude", 0)),
                })
            
            points.append(PointStruct(
                id=int(pid),
                vector=emb.tolist(),
                payload=payload,
            ))
        
        client.upsert(collection_name=collection_name, points=points)
    
    # Verify
    info = client.get_collection(collection_name)
    logger.info(f"Qdrant collection '{collection_name}': {info.points_count} points")
    
    return client


def export_query_tower(model, device):
    """
    Export the Query Tower as a standalone model for inference.
    Also export as TorchScript for deployment.
    """
    # Save PyTorch state dict
    query_tower_path = MODEL_DIR / "query_tower.pt"
    torch.save(model.query_tower.state_dict(), query_tower_path)
    logger.info(f"Query Tower state dict → {query_tower_path}")
    
    # Try TorchScript export
    try:
        model.query_tower.eval()
        dummy_text = torch.randn(1, TEXT_EMBED_DIM).to(device)
        dummy_eng = torch.randn(1, 4).to(device)
        
        traced = torch.jit.trace(model.query_tower, (dummy_text, dummy_eng))
        ts_path = MODEL_DIR / "query_tower_traced.pt"
        traced.save(str(ts_path))
        logger.info(f"Query Tower TorchScript → {ts_path}")
    except Exception as e:
        logger.warning(f"TorchScript export failed (non-critical): {e}")
    
    # Export item tower too (for re-indexing)
    item_tower_path = MODEL_DIR / "item_tower.pt"
    torch.save(model.item_tower.state_dict(), item_tower_path)
    logger.info(f"Item Tower state dict → {item_tower_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Two-Tower embeddings to Qdrant")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint (default: auto-detect best)")
    parser.add_argument("--graph_embeds", type=str, default=None,
                        help="Path to GNN place embeddings")
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION)
    parser.add_argument("--qdrant_host", type=str, default=QDRANT_HOST)
    parser.add_argument("--qdrant_port", type=int, default=QDRANT_PORT)
    parser.add_argument("--dry_run", action="store_true",
                        help="Compute embeddings only, skip Qdrant upload")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # ── Find model checkpoint ──
    if args.model is None:
        best_path = MODEL_DIR / "two_tower_best.pt"
        final_path = MODEL_DIR / "two_tower_final.pt"
        if best_path.exists():
            args.model = str(best_path)
        elif final_path.exists():
            args.model = str(final_path)
        else:
            logger.error("No model checkpoint found! Train first with 03_train_two_tower.py")
            sys.exit(1)
    
    logger.info(f"Using checkpoint: {args.model}")
    
    # ── Find GNN embeddings ──
    if args.graph_embeds is None:
        default_path = EMBED_DIR / "gnn_place_embeddings.npz"
        if default_path.exists():
            args.graph_embeds = str(default_path)
            logger.info(f"Auto-detected GNN embeddings: {args.graph_embeds}")
    
    # ── Load data ──
    places_df = pd.read_parquet(DATA_DIR / "places.parquet")
    logger.info(f"Loaded {len(places_df)} places")
    
    # Load item text embeddings
    item_cache = DATA_DIR / "item_text_embeds.npz"
    if item_cache.exists():
        logger.info("Loading cached item text embeddings...")
        item_data = np.load(item_cache)
        item_text_embeds = {int(k): item_data[k] for k in item_data.files}
    else:
        logger.info("Item text embeddings not cached. Encoding with bge-m3...")
        from sentence_transformers import SentenceTransformer
        from src.data.dataset import encode_texts_batched
        
        text_model = SentenceTransformer("BAAI/bge-m3")
        item_texts = []
        item_pids = []
        for _, row in places_df.iterrows():
            text_parts = [row["name"]]
            genres = row["wongnai_genres"]
            if isinstance(genres, (list, np.ndarray)) and len(genres) > 0:
                text_parts.extend([str(g) for g in genres])
            tags = row["tags"]
            if isinstance(tags, (list, np.ndarray)) and len(tags) > 0:
                text_parts.extend([str(t) for t in tags])
            intro = row["introduction"]
            if isinstance(intro, str) and intro:
                text_parts.append(intro[:200])
            item_texts.append(" ".join(text_parts))
            item_pids.append(row["place_id"])
        
        item_embeds_array = encode_texts_batched(item_texts, text_model, desc="Item texts")
        item_text_embeds = {pid: emb for pid, emb in zip(item_pids, item_embeds_array)}
        
        # Cache
        np.savez(item_cache, **{str(k): v for k, v in item_text_embeds.items()})
        del text_model
    
    # Load graph embeddings
    graph_embeds: dict[int, np.ndarray] = {}
    if args.graph_embeds and Path(args.graph_embeds).exists():
        logger.info(f"Loading GNN embeddings: {args.graph_embeds}")
        data = np.load(args.graph_embeds)
        for pid, emb in zip(data["place_ids"], data["embeddings"]):
            graph_embeds[int(pid)] = emb
        logger.info(f"  Loaded {len(graph_embeds)} GNN embeddings")
    
    # ── Load model ──
    model = load_model(args.model, device)
    
    # ── Build sequential ID maps (same as dataset.py) ──
    unique_cat_ids = sorted(places_df["category_id"].dropna().unique().tolist())
    unique_prov_ids = sorted(places_df["province_id"].dropna().unique().tolist())
    cat_id_map = {int(cid): i + 1 for i, cid in enumerate(unique_cat_ids)}
    prov_id_map = {int(pid): i + 1 for i, pid in enumerate(unique_prov_ids)}
    logger.info(f"  ID maps: {len(unique_cat_ids)} categories, {len(unique_prov_ids)} provinces")
    
    # ── Compute embeddings ──
    logger.info("Computing item embeddings for all places...")
    t_start = time.time()
    
    embeddings, place_ids = compute_item_embeddings(
        model, places_df, item_text_embeds, graph_embeds,
        cat_id_map, prov_id_map,
        device, batch_size=args.batch_size,
    )
    
    elapsed = time.time() - t_start
    logger.info(f"  Computed {embeddings.shape[0]} embeddings in {elapsed:.1f}s")
    logger.info(f"  Embedding shape: {embeddings.shape}")
    
    # Save local copy
    np.savez(
        EMBED_DIR / "two_tower_place_embeddings.npz",
        place_ids=np.array(place_ids),
        embeddings=embeddings,
    )
    logger.info(f"  Saved to {EMBED_DIR / 'two_tower_place_embeddings.npz'}")
    
    # ── Export Query Tower ──
    export_query_tower(model, device)
    
    # ── Upload to Qdrant ──
    if args.dry_run:
        logger.info("Dry run: skipping Qdrant upload")
    else:
        logger.info(f"Uploading to Qdrant ({args.qdrant_host}:{args.qdrant_port})...")
        try:
            upload_to_qdrant(
                place_ids, embeddings, places_df,
                collection_name=args.collection,
                host=args.qdrant_host,
                port=args.qdrant_port,
            )
            logger.info("Qdrant upload complete!")
        except Exception as e:
            logger.error(f"Qdrant upload failed: {e}")
            logger.error("You can retry with: python scripts/04_export_qdrant.py")
            logger.error("Or use --dry_run to just compute embeddings")
    
    # ── Summary ──
    logger.info("=" * 60)
    logger.info("Export complete!")
    logger.info(f"  Embeddings: {EMBED_DIR / 'two_tower_place_embeddings.npz'}")
    logger.info(f"  Query Tower: {MODEL_DIR / 'query_tower.pt'}")
    logger.info(f"  Item Tower: {MODEL_DIR / 'item_tower.pt'}")
    if not args.dry_run:
        logger.info(f"  Qdrant: {args.collection} @ {args.qdrant_host}:{args.qdrant_port}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
