#!/usr/bin/env python3
"""
Step 2: Train GNN on heterogeneous TikTok graph → export embeddings.

Run on GPU (vast.ai). Requires data/ folder from step 1.

Usage:
    python scripts/02_train_gnn.py [--epochs 50] [--lr 0.001]
"""

import argparse
import json
import logging
import sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_DIR, MODEL_DIR, EMBED_DIR,
    GNN_HIDDEN_DIM, GNN_EMBED_DIM, GNN_NUM_LAYERS,
    GNN_LEARNING_RATE, GNN_EPOCHS,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_GNN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train GNN on TikTok graph")
    parser.add_argument("--epochs", type=int, default=GNN_EPOCHS)
    parser.add_argument("--lr", type=float, default=GNN_LEARNING_RATE)
    parser.add_argument("--hidden_dim", type=int, default=GNN_HIDDEN_DIM)
    parser.add_argument("--out_dim", type=int, default=GNN_EMBED_DIM)
    parser.add_argument("--num_layers", type=int, default=GNN_NUM_LAYERS)
    parser.add_argument("--mlflow", action="store_true", default=True,
                        help="Enable MLflow tracking (default: True)")
    parser.add_argument("--no_mlflow", action="store_true",
                        help="Disable MLflow tracking")
    args = parser.parse_args()

    if args.no_mlflow:
        args.mlflow = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── MLflow ──
    from src.utils.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(
        experiment_name=MLFLOW_EXPERIMENT_GNN,
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=f"gnn-L{args.num_layers}-h{args.hidden_dim}-e{args.epochs}",
        params={
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "out_dim": args.out_dim,
            "num_layers": args.num_layers,
            "device": str(device),
        },
    ) if args.mlflow else None
    
    # ── Check PyG ──
    try:
        from torch_geometric.data import HeteroData
        from torch_geometric.nn import SAGEConv
    except ImportError:
        logger.error("torch-geometric not installed! Run:")
        logger.error("  pip install torch-geometric")
        sys.exit(1)
    
    # ── Load data ──
    import pandas as pd
    places_df = pd.read_parquet(DATA_DIR / "places.parquet")
    
    with open(DATA_DIR / "graph_data.json") as f:
        graph_data = json.load(f)
    
    with open(DATA_DIR / "video_features.json") as f:
        graph_data["video_features"] = json.load(f)
    
    logger.info(f"Loaded graph data: {len(graph_data['author_ids'])} authors, "
                f"{len(graph_data['video_ids'])} videos, {len(graph_data['hashtags'])} hashtags")
    
    # ── Build graph ──
    from src.models.gnn import build_hetero_graph, SimplifiedGNN, bpr_loss, sample_negative_places
    
    hetero_data, id_maps = build_hetero_graph(places_df, graph_data)
    hetero_data = hetero_data.to(device)
    
    n_places = len(id_maps["place_id_map"])
    n_authors = len(id_maps["author_id_map"])
    n_videos = len(id_maps["video_id_map"])
    n_hashtags = len(id_maps["hashtag_map"])
    
    # ── Build model (SimplifiedGNN — recommended starting point) ──
    model = SimplifiedGNN(
        n_places=n_places,
        n_authors=n_authors,
        n_videos=n_videos,
        n_hashtags=n_hashtags,
        place_feat_dim=hetero_data["place"].x.shape[1],
        video_feat_dim=hetero_data["video"].x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ── Training with BPR loss on video→place edges ──
    # Positive: video v is "about" place p (edge exists)
    # Negative: random place p' that v is NOT about
    
    vp_edges = hetero_data["video", "about", "place"].edge_index  # (2, n_edges)
    n_vp_edges = vp_edges.shape[1]
    
    logger.info(f"Training on {n_vp_edges} video→place edges")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}")
    
    best_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        # Forward: get all embeddings
        emb_dict = model(hetero_data)
        
        place_emb = emb_dict["place"]   # (n_places, out_dim)
        video_emb = emb_dict["video"]   # (n_videos, out_dim)
        
        # Positive scores: dot(video_emb[src], place_emb[dst])
        pos_video_idx = vp_edges[0]  # video indices
        pos_place_idx = vp_edges[1]  # place indices
        
        pos_scores = (video_emb[pos_video_idx] * place_emb[pos_place_idx]).sum(dim=-1)
        
        # Negative sampling: random places
        neg_place_idx = torch.randint(0, n_places, (n_vp_edges,), device=device)
        neg_scores = (video_emb[pos_video_idx] * place_emb[neg_place_idx]).sum(dim=-1)
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        # Regularization: prevent embedding collapse
        reg_loss = 0.001 * (place_emb.norm(dim=1).mean() + video_emb.norm(dim=1).mean())
        total_loss = loss + reg_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                # Quick metric: accuracy (pos_score > neg_score)
                acc = (pos_scores > neg_scores).float().mean().item()
                
                # Embedding stats
                place_norm = place_emb.norm(dim=1).mean().item()
                video_norm = video_emb.norm(dim=1).mean().item()
            
            logger.info(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Loss: {total_loss.item():.4f} (BPR: {loss.item():.4f}) | "
                f"Acc: {acc:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Norms: place={place_norm:.3f} video={video_norm:.3f}"
            )
            if tracker:
                tracker.log_metrics({
                    "loss": total_loss.item(),
                    "bpr_loss": loss.item(),
                    "accuracy": acc,
                    "lr": scheduler.get_last_lr()[0],
                    "place_norm": place_norm,
                    "video_norm": video_norm,
                }, step=epoch)
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(model.state_dict(), MODEL_DIR / "gnn_best.pt")
    
    # ── Export embeddings ──
    logger.info("Exporting embeddings...")
    model.load_state_dict(torch.load(MODEL_DIR / "gnn_best.pt", weights_only=True))
    model.eval()
    
    with torch.no_grad():
        emb_dict = model(hetero_data)
    
    # Place embeddings
    place_id_map_inv = {v: k for k, v in id_maps["place_id_map"].items()}
    place_ids_ordered = [place_id_map_inv[i] for i in range(n_places)]
    place_embeddings = emb_dict["place"].cpu().numpy()
    
    np.savez(
        EMBED_DIR / "gnn_place_embeddings.npz",
        place_ids=np.array(place_ids_ordered),
        embeddings=place_embeddings,
    )
    
    # Author embeddings
    author_id_map_inv = {v: k for k, v in id_maps["author_id_map"].items()}
    author_ids_ordered = [author_id_map_inv[i] for i in range(n_authors)]
    author_embeddings = emb_dict["author"].cpu().numpy()
    
    np.savez(
        EMBED_DIR / "gnn_author_embeddings.npz",
        author_ids=np.array(author_ids_ordered),
        embeddings=author_embeddings,
    )
    
    # Hashtag embeddings
    hashtag_map_inv = {v: k for k, v in id_maps["hashtag_map"].items()}
    hashtags_ordered = [hashtag_map_inv[i] for i in range(n_hashtags)]
    hashtag_embeddings = emb_dict["hashtag"].cpu().numpy()
    
    np.savez(
        EMBED_DIR / "gnn_hashtag_embeddings.npz",
        hashtags=np.array(hashtags_ordered),
        embeddings=hashtag_embeddings,
    )
    
    logger.info(f"  Place embeddings:   {place_embeddings.shape} → {EMBED_DIR / 'gnn_place_embeddings.npz'}")
    logger.info(f"  Author embeddings:  {author_embeddings.shape} → {EMBED_DIR / 'gnn_author_embeddings.npz'}")
    logger.info(f"  Hashtag embeddings: {hashtag_embeddings.shape} → {EMBED_DIR / 'gnn_hashtag_embeddings.npz'}")
    
    # ── Save place_to_authors and place_to_hashtags mappings (for re-ranker) ──
    logger.info("Building place→authors and place→hashtags mappings...")
    
    # Parse from graph_data edges
    place_to_authors: dict[int, list[str]] = {}
    place_to_hashtags: dict[int, list[str]] = {}
    
    # video → place edges
    video_to_places: dict[str, list[int]] = {}
    for v, p, _ in graph_data["video_place_edges"]:
        video_to_places.setdefault(v, []).append(p)
    
    # author → video → place
    for author_id, video_id in graph_data["author_video_edges"]:
        for place_id in video_to_places.get(video_id, []):
            place_to_authors.setdefault(place_id, []).append(author_id)
    
    # video → hashtag → video → place
    video_to_hashtags: dict[str, list[str]] = {}
    for video_id, hashtag in graph_data["video_hashtag_edges"]:
        video_to_hashtags.setdefault(video_id, []).append(hashtag)
    
    for video_id, hashtags in video_to_hashtags.items():
        for place_id in video_to_places.get(video_id, []):
            place_to_hashtags.setdefault(place_id, []).extend(hashtags)
    
    # Deduplicate
    place_to_authors = {k: list(set(v)) for k, v in place_to_authors.items()}
    place_to_hashtags = {k: list(set(v)) for k, v in place_to_hashtags.items()}
    
    with open(EMBED_DIR / "place_to_authors.json", "w") as f:
        json.dump({str(k): v for k, v in place_to_authors.items()}, f)
    
    with open(EMBED_DIR / "place_to_hashtags.json", "w") as f:
        json.dump({str(k): v for k, v in place_to_hashtags.items()}, f)
    
    logger.info(f"  place_to_authors:  {len(place_to_authors)} places")
    logger.info(f"  place_to_hashtags: {len(place_to_hashtags)} places")

    # ── Log artifacts to MLflow ──
    if tracker:
        tracker.log_artifact(MODEL_DIR / "gnn_best.pt", "models")
        tracker.log_artifact(EMBED_DIR / "gnn_place_embeddings.npz", "embeddings")
        tracker.log_artifact(EMBED_DIR / "gnn_author_embeddings.npz", "embeddings")
        tracker.log_artifact(EMBED_DIR / "gnn_hashtag_embeddings.npz", "embeddings")
        tracker.log_metrics({
            "best_loss": best_loss,
            "n_places": n_places,
            "n_authors": n_authors,
            "n_videos": n_videos,
            "n_hashtags": n_hashtags,
            "n_vp_edges": n_vp_edges,
        })
        tracker.end()

    logger.info("=" * 60)
    logger.info("GNN training complete!")
    logger.info(f"Next: python scripts/03_train_two_tower.py --graph_embeds {EMBED_DIR / 'gnn_place_embeddings.npz'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
