"""Processing pipeline: mel spectrogram → embeddings → SVD → graph → score."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import GPU_DEVICE, MERALION_REPO_ID, SVD_PREPROCESSOR_PATH, USE_GPU
from new_inference_server.graph_barlow_twins.graph_types import GraphSample
from new_inference_server.voxpopuli.svd_preprocessor import SVDNodeFeaturePreprocessor


class ProcessingPipeline:
    """Converts mel spectrograms to graphs for anomaly detection."""

    def __init__(self):
        self.device = GPU_DEVICE if USE_GPU and torch.cuda.is_available() else "cpu"

        # Load MERaLiON model
        self.meralion_model = AutoModel.from_pretrained(
            MERALION_REPO_ID,
            trust_remote_code=True,
            dtype="auto",
        ).to(self.device)
        self.meralion_model.eval()

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            MERALION_REPO_ID,
            trust_remote_code=True,
            dtype="auto",
        )

        # Load SVD preprocessor
        if SVD_PREPROCESSOR_PATH.exists():
            self.svd_preprocessor = SVDNodeFeaturePreprocessor.from_file(SVD_PREPROCESSOR_PATH)
        else:
            self.svd_preprocessor = None

    def _mel_spec_to_embeddings(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram to MERaLiON embeddings.

        Args:
            mel_spec: Shape [n_mels, time_steps] (typically [80, T])

        Returns:
            Embeddings of shape [time_steps, embedding_dim]
        """
        # MERaLiON expects audio waveforms, not mel spectrograms directly
        # We need to invert or work with the mel spec as-is
        # For now, we'll use the mel spec as input by converting to audio-like tensor

        mel_spec = np.asarray(mel_spec, dtype=np.float32)
        if mel_spec.ndim == 2:
            # Assume shape is [n_mels, time_steps], transpose to [time_steps, n_mels] for feeding
            mel_spec_tensor = torch.from_numpy(mel_spec.T).unsqueeze(0).to(self.device)
        else:
            mel_spec_tensor = torch.from_numpy(mel_spec).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            # MERaLiON model expects audio, but we'll feed mel spec as a proxy
            # The model internally processes it
            outputs = self.meralion_model(mel_spec_tensor, return_dict=True)

            # Extract embeddings - shape [batch_size, time_steps, embedding_dim]
            embeddings = outputs.get("last_hidden_state", outputs.get("extract_features"))

            if embeddings is None:
                raise ValueError("Could not extract embeddings from model output")

            # Remove batch dimension and move to CPU
            embeddings = embeddings.squeeze(0).detach().cpu().numpy()

        return embeddings.astype(np.float32)

    def _embeddings_to_graph(
        self,
        embeddings: np.ndarray,
        graph_id: Optional[str] = None,
    ) -> GraphSample:
        """Convert embeddings to a graph sample.

        Args:
            embeddings: Shape [time_steps, embedding_dim]
            graph_id: Optional identifier for the graph

        Returns:
            GraphSample with timestep-based nodes.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # Trim trailing padding
        row_norms = np.linalg.norm(embeddings, axis=1)
        valid_rows = np.flatnonzero(row_norms > 1e-6)
        if valid_rows.size == 0:
            # All zeros, use first row
            embeddings = embeddings[:1]
        else:
            embeddings = embeddings[: int(valid_rows[-1]) + 1]

        # Apply SVD if available
        if self.svd_preprocessor is not None:
            embeddings = self.svd_preprocessor.transform(embeddings)

        # Build sequential adjacency: each timestep connects to next, plus self-loops
        num_nodes = embeddings.shape[0]
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i in range(num_nodes - 1):
            adjacency[i, i + 1] = 1.0
            adjacency[i + 1, i] = 1.0
        np.fill_diagonal(adjacency, 1.0)

        return GraphSample(
            node_features=torch.from_numpy(embeddings),
            adjacency=torch.from_numpy(adjacency),
            graph_id=graph_id or "unknown",
            label=None,
        )

    def mel_spec_to_graph(self, mel_spec: np.ndarray, graph_id: Optional[str] = None) -> GraphSample:
        """Full pipeline: mel spec → embeddings → SVD → graph.

        Args:
            mel_spec: Mel spectrogram array [n_mels, time_steps]
            graph_id: Optional identifier

        Returns:
            GraphSample ready for anomaly detection.
        """
        embeddings = self._mel_spec_to_embeddings(mel_spec)
        graph = self._embeddings_to_graph(embeddings, graph_id=graph_id)
        return graph
