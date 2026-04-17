from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GraphSample:
    """A single graph input.

    Required format:
    - node_features: float tensor of shape [num_nodes, num_features]
    - adjacency: float tensor of shape [num_nodes, num_nodes]
      * values may be binary or weighted
      * self-loops are optional; the model adds them if absent
    Optional:
    - graph_id: identifier used in outputs
    - label: optional anomaly label for evaluation only
    """

    node_features: torch.Tensor
    adjacency: torch.Tensor
    graph_id: Optional[str] = None
    label: Optional[int] = None


def validate_graph_sample(graph: GraphSample) -> None:
    if not torch.is_tensor(graph.node_features):
        raise TypeError("node_features must be a torch.Tensor")
    if not torch.is_tensor(graph.adjacency):
        raise TypeError("adjacency must be a torch.Tensor")
    if graph.node_features.ndim != 2:
        raise ValueError("node_features must have shape [num_nodes, num_features]")
    if graph.adjacency.ndim != 2:
        raise ValueError("adjacency must have shape [num_nodes, num_nodes]")
    if graph.adjacency.shape[0] != graph.adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if graph.node_features.shape[0] != graph.adjacency.shape[0]:
        raise ValueError("node_features and adjacency must have the same number of nodes")
    if graph.node_features.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise TypeError("node_features must be floating point")
    if graph.adjacency.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise TypeError("adjacency must be floating point")
