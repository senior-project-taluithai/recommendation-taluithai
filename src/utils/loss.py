"""
Loss functions for Two-Tower training.
IPS-Weighted InfoNCE with temperature scaling.
"""

import torch
import torch.nn.functional as F


def ips_weighted_infonce_loss(
    query_emb: torch.Tensor,
    item_emb: torch.Tensor,
    propensity_scores: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss with Inverse Propensity Scoring for popularity debiasing.
    
    In-batch negatives: the i-th query is positive with i-th item,
    and negative with all other items in the batch.
    
    Args:
        query_emb: (batch_size, embed_dim) L2-normalized
        item_emb:  (batch_size, embed_dim) L2-normalized
        propensity_scores: (batch_size,) P_i for each positive item
        temperature: scaling factor for similarity scores
    
    Returns:
        scalar loss
    """
    # Similarity matrix: (batch, batch)
    sim_matrix = torch.matmul(query_emb, item_emb.T) / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    # Per-sample cross entropy loss
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction="none")
    
    # IPS weighting: weight_i = 1/P_i (unpopular → higher weight)
    ips_weights = 1.0 / propensity_scores.clamp(min=0.01)
    ips_weights = ips_weights / ips_weights.mean()  # normalize to mean=1
    
    return (loss_per_sample * ips_weights).mean()


def infonce_loss(
    query_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Standard InfoNCE without IPS (for comparison/ablation)."""
    sim_matrix = torch.matmul(query_emb, item_emb.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)
