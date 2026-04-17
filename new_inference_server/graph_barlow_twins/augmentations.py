from __future__ import annotations

import torch

from .graph_types import GraphSample


def add_self_loops(adjacency: torch.Tensor) -> torch.Tensor:
    n = adjacency.shape[0]
    eye = torch.eye(n, dtype=adjacency.dtype, device=adjacency.device)
    return adjacency + eye


def normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    adj = add_self_loops(adjacency)
    degree = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return adj / degree


def feature_mask(graph: GraphSample, mask_ratio: float = 0.2) -> GraphSample:
    x = graph.node_features.clone()
    n = x.shape[0]
    if n == 0:
        return graph
    num_mask = max(1, int(round(n * mask_ratio)))
    mask_idx = torch.randperm(n, device=x.device)[:num_mask]
    x[mask_idx] = 0.0
    return GraphSample(node_features=x, adjacency=graph.adjacency.clone(), graph_id=graph.graph_id, label=graph.label)


def edge_dropout(graph: GraphSample, dropout_ratio: float = 0.2) -> GraphSample:
    adj = graph.adjacency.clone()
    if adj.numel() == 0:
        return graph
    keep = torch.rand_like(adj) > dropout_ratio
    keep = torch.triu(keep, diagonal=1)
    keep = keep | keep.T
    adj = adj * keep.to(adj.dtype)
    return GraphSample(node_features=graph.node_features.clone(), adjacency=adj, graph_id=graph.graph_id, label=graph.label)


def gaussian_noise(graph: GraphSample, std: float = 0.01) -> GraphSample:
    x = graph.node_features.clone()
    x = x + torch.randn_like(x) * std
    return GraphSample(node_features=x, adjacency=graph.adjacency.clone(), graph_id=graph.graph_id, label=graph.label)
