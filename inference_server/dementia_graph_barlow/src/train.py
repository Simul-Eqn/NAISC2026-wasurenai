from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import TrainingConfig
from .data import DailyGraph, aggregation_feature_mask
from .model import GraphBarlowTwins, barlow_twins_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_graphs(graphs_by_household: Dict[str, List[DailyGraph]]) -> List[DailyGraph]:
    all_graphs: List[DailyGraph] = []
    for graphs in graphs_by_household.values():
        all_graphs.extend(graphs)
    return all_graphs


def train_graph_barlow(
    graphs: List[DailyGraph],
    config: TrainingConfig,
    device: torch.device,
) -> GraphBarlowTwins:
    if not graphs:
        raise ValueError("No graphs found for training")

    set_seed(config.seed)
    in_dim = graphs[0].x.size(-1)
    model = GraphBarlowTwins(
        in_dim=in_dim,
        hidden_dim=config.hidden_dim,
        projection_dim=config.projection_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.train()
    for _ in range(config.epochs):
        random.shuffle(graphs)
        for graph in graphs:
            x = graph.x.to(device)
            adj = graph.adj.to(device)

            x1 = aggregation_feature_mask(graph, config.mask_ratio, config.noise_std).to(device)
            x2 = aggregation_feature_mask(graph, config.mask_ratio, config.noise_std).to(device)

            _, z1, _ = model(x1, adj)
            _, z2, _ = model(x2, adj)
            loss = barlow_twins_loss(z1, z2, config.lambda_offdiag)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def encode_graphs(
    model: GraphBarlowTwins,
    graphs_by_household: Dict[str, List[DailyGraph]],
    device: torch.device,
) -> Dict[str, List[dict]]:
    model.eval()
    output: Dict[str, List[dict]] = {}

    for household_id, graphs in graphs_by_household.items():
        records = []
        for graph in graphs:
            embedding, _, node_weights = model(graph.x.to(device), graph.adj.to(device))
            records.append(
                {
                    "household_id": household_id,
                    "period_start": str(graph.period_start),
                    "day": str(graph.period_start),
                    "embedding": embedding.detach().cpu().numpy(),
                    "node_weights": node_weights.detach().cpu().numpy(),
                    "sensor_names": graph.sensor_names,
                }
            )
        output[household_id] = records

    return output


def save_checkpoint(model: GraphBarlowTwins, config: TrainingConfig, output_path: str | Path) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "training_config": asdict(config),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
