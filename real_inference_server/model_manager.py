"""Model manager for training and scoring personalized anomaly detection models."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import (
    GPU_DEVICE,
    MODELS_DIR,
    TRAINING_EPOCHS,
    TRAINING_LEARNING_RATE,
    THRESHOLD_QUANTILE,
    USE_GPU,
)
from new_inference_server.graph_barlow_twins import (
    GraphAnomalyConfig,
    GraphBarlowAnomalyDetector,
)
from new_inference_server.graph_barlow_twins.graph_types import GraphSample
from new_inference_server.graph_barlow_twins.model import barlow_twins_loss


class ModelManager:
    """Manages training and inference of personalized anomaly models."""

    def __init__(self):
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = GPU_DEVICE if USE_GPU and torch.cuda.is_available() else "cpu"

        # In-memory model cache: chat_id -> GraphBarlowAnomalyDetector
        self.model_cache: dict[int, GraphBarlowAnomalyDetector] = {}
        self.thresholds: dict[int, float] = {}

        # Global config
        self.config = GraphAnomalyConfig(
            hidden_dim=64,
            projection_dim=64,
            num_layers=2,
            dropout=0.1,
            learning_rate=TRAINING_LEARNING_RATE,
            epochs=TRAINING_EPOCHS,
        )

    def _get_model_path(self, chat_id: int) -> Path:
        """Get the model save path for a user."""
        return self.models_dir / f"model_{chat_id}.pt"

    def _get_threshold_path(self, chat_id: int) -> Path:
        """Get the threshold save path for a user."""
        return self.models_dir / f"threshold_{chat_id}.npy"

    def _to_device(self, graph: GraphSample) -> GraphSample:
        """Move a graph to the device."""
        return GraphSample(
            node_features=graph.node_features.to(self.device),
            adjacency=graph.adjacency.to(self.device),
            graph_id=graph.graph_id,
            label=graph.label,
        )

    def train_model(self, chat_id: int, graphs: list[GraphSample]) -> tuple[float, float]:
        """Train a personalized anomaly detection model.

        Args:
            chat_id: User identifier
            graphs: List of GraphSample objects for training

        Returns:
            Tuple of (threshold, mean_score)
        """
        if not graphs:
            raise ValueError("No graphs provided for training")

        # Infer input dimension from first graph
        in_dim = int(graphs[0].node_features.shape[1])

        # Create model
        model = GraphBarlowAnomalyDetector(in_dim=in_dim, config=self.config).to(self.device)

        # Train with self-supervision
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for graph in graphs:
                graph_device = self._to_device(graph)
                aug1 = model.augment(graph_device)
                aug2 = model.augment(graph_device)
                _, z1, _ = model(aug1)
                _, z2, _ = model(aug2)
                loss = barlow_twins_loss(z1, z2, self.config.lambda_offdiag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

        # Fit center on training data
        model.fit_center(graphs, device=self.device)

        # Compute scores and threshold
        scores = []
        model.eval()
        with torch.inference_mode():
            for graph in graphs:
                graph_device = self._to_device(graph)
                score = model.score(graph_device, device=self.device)
                scores.append(float(score))

        threshold = float(np.quantile(scores, THRESHOLD_QUANTILE))
        mean_score = float(np.mean(scores))

        # Cache model and threshold
        self.model_cache[chat_id] = model
        self.thresholds[chat_id] = threshold

        # Save to disk
        self._save_model(chat_id, model, threshold)

        return threshold, mean_score

    def _save_model(self, chat_id: int, model: GraphBarlowAnomalyDetector, threshold: float) -> None:
        """Save model to disk."""
        model_path = self._get_model_path(chat_id)
        threshold_path = self._get_threshold_path(chat_id)

        torch.save(model.state_dict(), model_path)
        np.save(threshold_path, np.array(threshold, dtype=np.float32))

    def _load_model(self, chat_id: int, in_dim: int) -> Optional[GraphBarlowAnomalyDetector]:
        """Load model from disk."""
        if chat_id in self.model_cache:
            return self.model_cache[chat_id]

        model_path = self._get_model_path(chat_id)
        threshold_path = self._get_threshold_path(chat_id)

        if not model_path.exists() or not threshold_path.exists():
            return None

        model = GraphBarlowAnomalyDetector(in_dim=in_dim, config=self.config).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        threshold = float(np.load(threshold_path))
        self.model_cache[chat_id] = model
        self.thresholds[chat_id] = threshold

        return model

    def score_graph(self, chat_id: int, graph: GraphSample) -> tuple[float, str]:
        """Score a single graph and return anomaly level.

        Args:
            chat_id: User identifier
            graph: GraphSample to score

        Returns:
            Tuple of (anomaly_score, anomaly_level) where level is "normal", "mild_anomaly", "high_anomaly"
        """
        in_dim = int(graph.node_features.shape[1])

        # Load model
        model = self._load_model(chat_id, in_dim)
        if model is None:
            raise ValueError(f"No trained model for user {chat_id}")

        threshold = self.thresholds.get(chat_id)
        if threshold is None:
            raise ValueError(f"No threshold for user {chat_id}")

        # Score
        graph_device = self._to_device(graph)
        with torch.inference_mode():
            score = model.score(graph_device, device=self.device)

        # Classify into levels
        # Normal: score <= threshold
        # Mild anomaly: threshold < score <= threshold * 1.5
        # High anomaly: score > threshold * 1.5
        score_float = float(score)
        if score_float <= threshold:
            level = "normal"
        elif score_float <= threshold * 1.5:
            level = "mild_anomaly"
        else:
            level = "high_anomaly"

        return score_float, level

    def has_model(self, chat_id: int) -> bool:
        """Check if a user has a trained model."""
        return self._get_model_path(chat_id).exists()
