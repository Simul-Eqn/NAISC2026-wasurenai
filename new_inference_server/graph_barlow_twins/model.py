from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentations import edge_dropout, feature_mask, gaussian_noise, normalize_adjacency
from .graph_types import GraphSample, validate_graph_sample


@dataclass
class GraphAnomalyConfig:
    hidden_dim: int = 64
    projection_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    mask_ratio: float = 0.2
    edge_dropout_ratio: float = 0.1
    noise_std: float = 0.01
    lambda_offdiag: float = 0.005
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adj = normalize_adjacency(adjacency)
        h = torch.matmul(adj, x)
        h = self.linear(h)
        h = F.relu(h)
        h = self.dropout(h)
        return self.norm(h)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [GraphConvolution(dims[i], dims[i + 1], dropout) for i in range(num_layers)]
        )
        self.graph_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for layer in self.layers:
            h = layer(h, adjacency)

        gate = torch.softmax(self.graph_gate(h).squeeze(-1), dim=0)
        graph_embedding = torch.sum(gate.unsqueeze(-1) * h, dim=0)
        return graph_embedding, gate


class GraphBarlowAnomalyDetector(nn.Module):
    """Graph Barlow Twins variant for anomaly detection on input graphs.

    Input graph format:
    - GraphSample.node_features: [num_nodes, num_features] float tensor
    - GraphSample.adjacency: [num_nodes, num_nodes] float tensor
    - optional graph_id/label
    """

    def __init__(self, in_dim: int, config: GraphAnomalyConfig):
        super().__init__()
        self.config = config
        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.projection_dim),
        )
        self.register_buffer("center", torch.zeros(config.hidden_dim))
        self.register_buffer("center_initialized", torch.tensor(False))

    def forward(self, graph: GraphSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        validate_graph_sample(graph)
        embedding, node_weights = self.encoder(graph.node_features, graph.adjacency)
        projection = self.projector(embedding.unsqueeze(0)).squeeze(0)
        return embedding, projection, node_weights

    def augment(self, graph: GraphSample) -> GraphSample:
        g = feature_mask(graph, self.config.mask_ratio)
        g = edge_dropout(g, self.config.edge_dropout_ratio)
        g = gaussian_noise(g, self.config.noise_std)
        return g

    @torch.no_grad()
    def fit_center(self, graphs: list[GraphSample], device: torch.device) -> None:
        self.eval()
        embeddings = []
        for graph in graphs:
            emb, _, _ = self.forward(GraphSample(graph.node_features.to(device), graph.adjacency.to(device), graph.graph_id, graph.label))
            embeddings.append(emb.detach())
        if not embeddings:
            raise ValueError("No graphs provided to fit_center")
        self.center.copy_(torch.stack(embeddings, dim=0).mean(dim=0))
        self.center_initialized.fill_(True)

    @torch.no_grad()
    def score(self, graph: GraphSample, device: torch.device) -> float:
        if not bool(self.center_initialized.item()):
            raise RuntimeError("Center not initialized. Call fit_center() first.")
        self.eval()
        emb, _, _ = self.forward(GraphSample(graph.node_features.to(device), graph.adjacency.to(device), graph.graph_id, graph.label))
        return float(1.0 - F.cosine_similarity(emb, self.center, dim=0).item())


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_offdiag: float) -> torch.Tensor:
    z1 = (z1 - z1.mean()) / (z1.std(unbiased=False) + 1e-9)
    z2 = (z2 - z2.mean()) / (z2.std(unbiased=False) + 1e-9)
    c = torch.outer(z1, z2) / z1.numel()
    on_diag = torch.diagonal(c).add(-1.0).pow(2).sum()
    off_diag_mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
    off_diag = c[off_diag_mask].pow(2).sum()
    return on_diag + lambda_offdiag * off_diag
