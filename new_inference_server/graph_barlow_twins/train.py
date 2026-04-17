from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch

from .graph_types import GraphSample
from .model import GraphAnomalyConfig, GraphBarlowAnomalyDetector, barlow_twins_loss


@dataclass
class TrainingResult:
    model: GraphBarlowAnomalyDetector
    losses: List[float]


def fit_graph_barlow_anomaly(
    graphs: Iterable[GraphSample],
    config: GraphAnomalyConfig,
    device: torch.device | None = None,
) -> TrainingResult:
    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("graphs must not be empty")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = graph_list[0].node_features.shape[1]
    model = GraphBarlowAnomalyDetector(in_dim=in_dim, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    losses: List[float] = []
    model.train()
    for _epoch in range(config.epochs):
        epoch_loss = 0.0
        for graph in graph_list:
            graph = GraphSample(
                node_features=graph.node_features.to(device),
                adjacency=graph.adjacency.to(device),
                graph_id=graph.graph_id,
                label=graph.label,
            )
            aug1 = model.augment(graph)
            aug2 = model.augment(graph)
            _, z1, _ = model(aug1)
            _, z2, _ = model(aug2)
            loss = barlow_twins_loss(z1, z2, config.lambda_offdiag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, len(graph_list)))

    model.fit_center(graph_list, device=device)
    return TrainingResult(model=model, losses=losses)


@torch.no_grad()
def score_graphs(model: GraphBarlowAnomalyDetector, graphs: Iterable[GraphSample], device: torch.device | None = None) -> list[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    for graph in graphs:
        score = model.score(GraphSample(graph.node_features.to(device), graph.adjacency.to(device), graph.graph_id, graph.label), device=device)
        results.append({"graph_id": graph.graph_id, "score": score, "label": graph.label})
    return results
