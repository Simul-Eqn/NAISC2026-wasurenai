from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        n_nodes = h.size(0)

        h_i = h.unsqueeze(1).expand(n_nodes, n_nodes, h.size(-1))
        h_j = h.unsqueeze(0).expand(n_nodes, n_nodes, h.size(-1))
        scores = self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)

        mask = adj > 0
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(scores, dim=1)
        alpha = self.dropout(alpha)

        out = torch.matmul(alpha, h)
        return F.elu(out), alpha


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            layers.append(GraphAttentionLayer(dims[i], dims[i + 1], dropout))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.node_gate = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_last = None
        h = x
        for layer in self.layers:
            h, attn_last = layer(h, adj)

        h = self.norm(h)
        node_scores = self.node_gate(h).squeeze(-1)
        node_weights = F.softmax(node_scores, dim=0)
        graph_embedding = torch.sum(node_weights.unsqueeze(-1) * h, dim=0)
        return graph_embedding, node_weights if attn_last is None else node_weights


class GraphBarlowTwins(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, projection_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = GraphEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, node_weights = self.encoder(x, adj)
        projection = self.projector(embedding.unsqueeze(0)).squeeze(0)
        return embedding, projection, node_weights


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_offdiag: float) -> torch.Tensor:
    z1 = (z1 - z1.mean()) / (z1.std(unbiased=False) + 1e-9)
    z2 = (z2 - z2.mean()) / (z2.std(unbiased=False) + 1e-9)

    c = torch.outer(z1, z2) / z1.numel()
    on_diag = torch.diagonal(c).add(-1.0).pow(2).sum()

    off_diag_mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
    off_diag = c[off_diag_mask].pow(2).sum()
    return on_diag + lambda_offdiag * off_diag
