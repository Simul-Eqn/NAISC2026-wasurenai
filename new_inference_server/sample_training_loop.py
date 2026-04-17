from __future__ import annotations

import torch

from .graph_barlow_twins.graph_types import GraphSample
from .graph_barlow_twins.model import GraphAnomalyConfig
from .graph_barlow_twins.train import fit_graph_barlow_anomaly, score_graphs


def build_toy_graphs() -> list[GraphSample]:
    graphs = []
    for i in range(8):
        n = 6
        x = torch.randn(n, 16)
        adj = torch.zeros(n, n)
        for j in range(n - 1):
            adj[j, j + 1] = 1.0
            adj[j + 1, j] = 1.0
        if i >= 6:
            x = x + 3.0 * torch.randn_like(x)
        graphs.append(GraphSample(node_features=x, adjacency=adj, graph_id=f"g{i}", label=1 if i >= 6 else 0))
    return graphs


def main() -> None:
    graphs = build_toy_graphs()
    config = GraphAnomalyConfig(epochs=20, hidden_dim=32, projection_dim=32)
    result = fit_graph_barlow_anomaly(graphs[:6], config=config)
    scores = score_graphs(result.model, graphs[6:])
    print("losses:", result.losses[-5:])
    print("scores:", scores)


if __name__ == "__main__":
    main()
