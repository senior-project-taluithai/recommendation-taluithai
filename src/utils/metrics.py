"""
Evaluation metrics for retrieval quality.
Recall@K, NDCG@K, Hit Rate.
"""

import torch
import numpy as np
from typing import Optional


def recall_at_k(
    query_embs: torch.Tensor,    # (n_queries, dim)
    item_embs: torch.Tensor,     # (n_items, dim)
    labels: torch.Tensor,        # (n_queries,) index of correct item in item_embs
    k_values: list[int] = [10, 50, 100],
) -> dict[str, float]:
    """
    Compute Recall@K: fraction of queries where the correct item
    appears in the top-K retrieved items.
    
    This is computed in batches to avoid OOM on large item sets.
    """
    device = query_embs.device
    n_queries = query_embs.shape[0]
    results = {}
    
    max_k = max(k_values)
    
    # Compute similarity in batches
    batch_size = 512
    all_topk_indices = []
    
    for i in range(0, n_queries, batch_size):
        q_batch = query_embs[i:i + batch_size]
        sim = torch.matmul(q_batch, item_embs.T)  # (batch, n_items)
        _, topk_idx = sim.topk(max_k, dim=-1)  # (batch, max_k)
        all_topk_indices.append(topk_idx)
    
    topk_indices = torch.cat(all_topk_indices, dim=0)  # (n_queries, max_k)
    
    for k in k_values:
        topk_k = topk_indices[:, :k]  # (n_queries, k)
        hits = (topk_k == labels.unsqueeze(1)).any(dim=1).float()
        results[f"recall@{k}"] = hits.mean().item()
    
    return results


def ndcg_at_k(
    query_embs: torch.Tensor,
    item_embs: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> float:
    """
    NDCG@K (simplified for single relevant item per query).
    """
    sim = torch.matmul(query_embs, item_embs.T)
    _, topk_idx = sim.topk(k, dim=-1)
    
    # Position of correct item (1-indexed)
    match = (topk_idx == labels.unsqueeze(1))
    positions = match.float().argmax(dim=1) + 1  # (n_queries,)
    found = match.any(dim=1).float()
    
    # DCG = 1/log2(pos+1) if found, else 0
    dcg = found / torch.log2(positions.float() + 1)
    
    # IDCG = 1/log2(2) = 1.0 (best case: correct item at position 1)
    idcg = 1.0
    
    return (dcg / idcg).mean().item()


@torch.no_grad()
def evaluate_retrieval(
    model,
    val_loader,
    all_item_embs: torch.Tensor,   # (n_places, 128) pre-computed
    place_id_to_idx: dict[int, int],
    device: torch.device,
    k_values: list[int] = [10, 50, 100],
) -> dict[str, float]:
    """
    Full retrieval evaluation on validation set.
    For each val query, check if the correct place is in top-K among ALL places.
    """
    model.eval()
    
    all_query_embs = []
    all_labels = []
    
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        query_emb = model.query_tower(
            batch["query_text_emb"],
            batch["query_engagement"],
        )
        all_query_embs.append(query_emb.cpu())
        
        # Map place_id to index in all_item_embs
        place_ids = batch["place_id"]
        if isinstance(place_ids, torch.Tensor):
            place_ids = place_ids.tolist()
        labels = torch.tensor([place_id_to_idx.get(pid, 0) for pid in place_ids])
        all_labels.append(labels)
    
    query_embs = torch.cat(all_query_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    metrics = recall_at_k(query_embs, all_item_embs.cpu(), labels, k_values)
    metrics["ndcg@10"] = ndcg_at_k(query_embs, all_item_embs.cpu(), labels, k=10)
    
    return metrics
