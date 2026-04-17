# new_inference_server: Graph Barlow Anomaly

This package is importable by other Python files and implements a Graph Barlow Twins style model for anomaly detection directly on graphs.

## Required graph format

Use this dataclass:

- `GraphSample.node_features`: `torch.Tensor` with shape `[num_nodes, num_features]`
- `GraphSample.adjacency`: `torch.Tensor` with shape `[num_nodes, num_nodes]`
- `GraphSample.graph_id`: optional string
- `GraphSample.label`: optional int for evaluation only

### Rules

- `node_features` and `adjacency` must have the same number of nodes.
- `adjacency` may be weighted or binary.
- Self-loops are optional; the model adds them during normalization.
- No PyG/DGL required.

## What the model does

- Creates two augmented views of each input graph.
- Encodes each graph with a simple dense graph encoder.
- Uses Barlow Twins loss to align the two views.
- Fits a center embedding from training graphs.
- Scores anomalies by cosine distance from that center.

## Import example

```python
from inference_server.new_inference_server import GraphSample, GraphAnomalyConfig, fit_graph_barlow_anomaly, score_graphs
```

## Sample training loop

See `sample_training_loop.py`.

### Run

```bash
python -m inference_server.new_inference_server.sample_training_loop
```
