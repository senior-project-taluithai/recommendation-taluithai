#!/usr/bin/env python3
"""
Step 3: Train Two-Tower (Query-Item) retrieval model.

Run on GPU (vast.ai). Requires:
  - data/ folder from step 1
  - (optional) embeddings/gnn_place_embeddings.npz from step 2

Usage:
    python scripts/03_train_two_tower.py
    python scripts/03_train_two_tower.py --graph_embeds embeddings/gnn_place_embeddings.npz
    python scripts/03_train_two_tower.py --epochs 20 --lr 5e-5 --use_ips
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    DATA_DIR, MODEL_DIR, EMBED_DIR,
    EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, TEMPERATURE,
    NUM_CATEGORIES, NUM_PROVINCES, NUM_REGIONS,
    GNN_EMBED_DIM, TEXT_EMBED_DIM, RECALL_K,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_TWO_TOWER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def pre_compute_all_item_embs(
    model, places_df, item_text_embeds, graph_embeds,
    cat_id_map, prov_id_map,
    device, batch_size=512,
) -> tuple[torch.Tensor, dict[int, int]]:
    """
    Pre-compute item embeddings for ALL places.
    Used for Recall@K evaluation (search among all 20K places).
    
    Returns:
        all_item_embs: (n_places, 128) tensor
        place_id_to_idx: dict mapping place_id → index in all_item_embs
    """
    model.eval()
    
    place_ids = places_df["place_id"].tolist()
    place_id_to_idx = {pid: i for i, pid in enumerate(place_ids)}
    
    all_embs = []
    
    for i in range(0, len(place_ids), batch_size):
        batch_pids = place_ids[i:i + batch_size]
        
        # Text embeddings
        text_embs = torch.stack([
            torch.tensor(item_text_embeds.get(pid, np.zeros(TEXT_EMBED_DIM)), dtype=torch.float32)
            for pid in batch_pids
        ]).to(device)
        
        # Categorical features (use sequential ID maps, matching dataset.py)
        cats = []
        nums = []
        for pid in batch_pids:
            row = places_df[places_df["place_id"] == pid].iloc[0]
            raw_cat = int(row["category_id"]) if not np.isnan(row["category_id"]) else 0
            raw_prov = int(row["province_id"]) if not np.isnan(row["province_id"]) else 0
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
        
        # Graph embeddings
        g_embs = torch.stack([
            torch.tensor(graph_embeds.get(pid, np.zeros(128)), dtype=torch.float32)
            for pid in batch_pids
        ]).to(device)
        
        with torch.no_grad():
            emb = model.encode_item(text_embs, cat_tensor, num_tensor, g_embs)
        all_embs.append(emb.cpu())
    
    return torch.cat(all_embs, dim=0), place_id_to_idx


def _region_to_id(region: str) -> int:
    mapping = {
        "North": 1, "Central": 2,
        "Northeast": 3, "West": 4,
        "East": 5, "South": 6,
    }
    return mapping.get(region, 0)


def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower Retrieval Model")
    parser.add_argument("--graph_embeds", type=str, default=None,
                        help="Path to GNN place embeddings (.npz)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--use_ips", action="store_true", default=True,
                        help="Use IPS-weighted InfoNCE (default: True)")
    parser.add_argument("--no_ips", action="store_true",
                        help="Disable IPS weighting")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="taluithai-rec")
    parser.add_argument("--mlflow", action="store_true", default=True,
                        help="Enable MLflow tracking (default: True)")
    parser.add_argument("--no_mlflow", action="store_true",
                        help="Disable MLflow tracking")
    args = parser.parse_args()

    if args.no_ips:
        args.use_ips = False
    if args.no_mlflow:
        args.mlflow = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ── Auto-detect GNN embeddings ──
    if args.graph_embeds is None:
        default_path = EMBED_DIR / "gnn_place_embeddings.npz"
        if default_path.exists():
            args.graph_embeds = str(default_path)
            logger.info(f"Auto-detected GNN embeddings: {args.graph_embeds}")
    
    # ── W&B ──
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"two-tower-{'ips' if args.use_ips else 'base'}-e{args.epochs}",
            )
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
            args.wandb = False

    # ── MLflow ──
    from src.utils.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(
        experiment_name=MLFLOW_EXPERIMENT_TWO_TOWER,
        tracking_uri=MLFLOW_TRACKING_URI,
        run_name=f"two-tower-{'ips' if args.use_ips else 'base'}-e{args.epochs}-lr{args.lr}",
        params={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "use_ips": args.use_ips,
            "hidden_dim": HIDDEN_DIM,
            "embed_dim": EMBEDDING_DIM,
            "has_graph_embeds": args.graph_embeds is not None,
            "device": str(device),
        },
    ) if args.mlflow else None
    
    # ── Prepare data ──
    from src.data.dataset import prepare_data
    
    # Temporarily override batch size
    import config
    original_batch = config.BATCH_SIZE
    config.BATCH_SIZE = args.batch_size
    
    train_loader, val_loader, metadata = prepare_data(
        graph_embeds_path=args.graph_embeds,
    )
    
    config.BATCH_SIZE = original_batch
    
    logger.info(f"Train: {metadata['n_train']} samples, Val: {metadata['n_val']} samples")
    logger.info(f"Places: {metadata['n_places']}, Has GNN embeds: {metadata['has_graph_embeds']}")
    
    # ── Build model ──
    from src.models.two_tower import TwoTowerModel
    
    model = TwoTowerModel(
        text_dim=TEXT_EMBED_DIM,
        engagement_dim=4,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBEDDING_DIM,
        numerical_dim=5,
        graph_dim=GNN_EMBED_DIM,
        cat_embed_dim=16,
        dropout=0.1,
        num_categories=metadata["num_categories"],
        num_provinces=metadata["num_provinces"],
        num_regions=metadata["num_regions"],
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )
    
    # ── Loss function ──
    from src.utils.loss import ips_weighted_infonce_loss, infonce_loss
    from src.utils.metrics import recall_at_k, ndcg_at_k, evaluate_retrieval
    
    loss_fn_name = "IPS-InfoNCE" if args.use_ips else "InfoNCE"
    logger.info(f"Loss: {loss_fn_name}, Temperature: {args.temperature}")
    
    # ── Training loop ──
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info("=" * 60)
    
    best_recall_50 = 0.0
    best_epoch = 0
    train_losses = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # Forward pass
            query_emb, item_emb = model(batch)
            
            # Loss
            if args.use_ips:
                loss = ips_weighted_infonce_loss(
                    query_emb, item_emb,
                    batch["propensity"],
                    temperature=args.temperature,
                )
            else:
                loss = infonce_loss(
                    query_emb, item_emb,
                    temperature=args.temperature,
                )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_steps += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / max(epoch_steps, 1)
        train_losses.append(avg_loss)
        elapsed = time.time() - t_start
        
        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Log epoch-level training metrics to MLflow
        if tracker:
            tracker.log_metrics({
                "train_loss": avg_loss,
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)
        
        # ── Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.info("  Evaluating on validation set...")
            
            # Pre-compute all item embeddings
            all_item_embs, place_id_to_idx = pre_compute_all_item_embs(
                model,
                metadata["places_df"],
                metadata["item_text_embeds"],
                metadata["graph_embeds"],
                metadata["cat_id_map"],
                metadata["prov_id_map"],
                device,
            )
            
            metrics = evaluate_retrieval(
                model, val_loader, all_item_embs,
                place_id_to_idx, device, k_values=RECALL_K,
            )
            
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"  Metrics: {metrics_str}")
            
            # Best model checkpoint (by recall@50)
            r50 = metrics.get("recall@50", 0)
            if r50 > best_recall_50:
                best_recall_50 = r50
                best_epoch = epoch
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                    "num_categories": metadata["num_categories"],
                    "num_provinces": metadata["num_provinces"],
                    "num_regions": metadata["num_regions"],
                }, MODEL_DIR / "two_tower_best.pt")
                
                logger.info(f"  ★ New best model saved! Recall@50={r50:.4f}")
            
            # MLflow: log val metrics
            if tracker:
                tracker.log_metrics(metrics, step=epoch)

            # W&B logging
            if args.wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    **metrics,
                })
        elif args.wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
    
    # ── Final save ──
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "num_categories": metadata["num_categories"],
        "num_provinces": metadata["num_provinces"],
        "num_regions": metadata["num_regions"],
    }, MODEL_DIR / "two_tower_final.pt")
    
    # ── Summary ──
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Best epoch: {best_epoch}, Recall@50: {best_recall_50:.4f}")
    logger.info(f"  Best model: {MODEL_DIR / 'two_tower_best.pt'}")
    logger.info(f"  Final model: {MODEL_DIR / 'two_tower_final.pt'}")
    logger.info("")
    logger.info(f"Next: python scripts/04_export_qdrant.py")
    logger.info("=" * 60)

    # ── MLflow: log final metrics and artifacts ──
    if tracker:
        tracker.log_metrics({
            "best_recall_50": best_recall_50,
            "best_epoch": best_epoch,
            "total_params": total_params,
            "n_train": metadata["n_train"],
            "n_val": metadata["n_val"],
            "n_places": metadata["n_places"],
        })
        tracker.log_artifact(MODEL_DIR / "two_tower_best.pt", "models")
        tracker.log_artifact(MODEL_DIR / "two_tower_final.pt", "models")
        tracker.end()

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
