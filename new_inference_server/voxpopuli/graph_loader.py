from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np
import torch

from ..graph_barlow_twins.graph_types import GraphSample
from .loader import load_speaker_manifest
from .svd_preprocessor import SVDNodeFeaturePreprocessor


DEFAULT_SVD_PREPROCESSOR_PATH = Path(__file__).resolve().with_name("svd_preprocessor.npz")


def _trim_trailing_padding(audio_embedding: Any, pad_atol: float = 1e-6) -> np.ndarray:
    """Remove trailing padded timesteps from a fixed-length embedding matrix.

    Expected raw shape in the JSON files:
    - [max_timesteps_in_batch, feature_dim]

    The loader assumes padding is appended at the end.
    """

    arr = np.asarray(audio_embedding, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"audio_embedding must be 1D or 2D, got shape {arr.shape}")

    row_norms = np.linalg.norm(arr, axis=1)
    valid_rows = np.flatnonzero(row_norms > pad_atol)
    if valid_rows.size == 0:
        return arr[:1]

    last_valid_row = int(valid_rows[-1]) + 1
    return arr[:last_valid_row]


def _resolve_preprocessor(
    use_svd_preprocessor: bool,
    svd_preprocessor: Optional[SVDNodeFeaturePreprocessor] = None,
    svd_preprocessor_path: str | Path | None = None,
) -> Optional[SVDNodeFeaturePreprocessor]:
    if not use_svd_preprocessor:
        return None
    if svd_preprocessor is not None:
        return svd_preprocessor
    if svd_preprocessor_path is None:
        svd_preprocessor_path = DEFAULT_SVD_PREPROCESSOR_PATH
    return SVDNodeFeaturePreprocessor.from_file(svd_preprocessor_path)


def _record_to_graph(
    record: dict[str, Any],
    svd_preprocessor: Optional[SVDNodeFeaturePreprocessor] = None,
) -> GraphSample:
    """Convert one utterance JSON record into one graph.

    Graph structure:
    - nodes = timesteps in the utterance
    - node features = timestep embeddings (padding removed)
    - edges = sequential adjacency between consecutive timesteps + self-loops
    """

    emb = _trim_trailing_padding(record.get("audio_embedding", []))
    if emb.shape[0] == 0:
        raise ValueError(f"Empty embedding after trimming padding for audio_id={record.get('audio_id')}")

    if svd_preprocessor is not None:
        emb = svd_preprocessor.transform(emb)

    adjacency = _build_sequential_adjacency(emb.shape[0])

    has_accent = record.get("has_accent")
    label: Optional[int]
    if has_accent is True:
        label = 1
    elif has_accent is False:
        label = 0
    else:
        label = None

    return GraphSample(
        node_features=torch.from_numpy(emb),
        adjacency=torch.from_numpy(adjacency),
        graph_id=str(record.get("audio_id") or record.get("speaker_id") or "unknown_audio"),
        label=label,
    )


def _build_sequential_adjacency(num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for idx in range(num_nodes - 1):
        adj[idx, idx + 1] = 1.0
        adj[idx + 1, idx] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def load_speaker_graph_records(data_root: str | Path, speaker_id: str) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    speaker_manifest = load_speaker_manifest(data_root, speaker_id)
    speaker_dir = data_root / speaker_id
    records: list[dict[str, Any]] = []

    for item in speaker_manifest.get("files", []):
        record_path = speaker_dir / item["file"]
        with record_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        records.append(record)

    return records


def build_utterance_graph(
    data_root: str | Path,
    speaker_id: str,
    file_name: Optional[str] = None,
    use_svd_preprocessor: bool = False,
    svd_preprocessor: Optional[SVDNodeFeaturePreprocessor] = None,
    svd_preprocessor_path: str | Path | None = None,
) -> GraphSample:
    """Load one utterance JSON and convert it into a timestep graph."""

    data_root = Path(data_root)
    resolved_preprocessor = _resolve_preprocessor(
        use_svd_preprocessor=use_svd_preprocessor,
        svd_preprocessor=svd_preprocessor,
        svd_preprocessor_path=svd_preprocessor_path,
    )
    if file_name is not None:
        record_path = data_root / speaker_id / file_name
        with record_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        return _record_to_graph(record, svd_preprocessor=resolved_preprocessor)

    records = load_speaker_graph_records(data_root, speaker_id)
    if not records:
        raise ValueError(f"No records found for speaker {speaker_id}")
    # Default to the first record when no filename is provided.
    return _record_to_graph(records[0], svd_preprocessor=resolved_preprocessor)


def load_speaker_graphs(
    data_root: str | Path,
    speaker_id: str,
    use_svd_preprocessor: bool = False,
    svd_preprocessor: Optional[SVDNodeFeaturePreprocessor] = None,
    svd_preprocessor_path: str | Path | None = None,
) -> list[GraphSample]:
    """Load all utterance graphs for a single speaker."""

    data_root = Path(data_root)
    resolved_preprocessor = _resolve_preprocessor(
        use_svd_preprocessor=use_svd_preprocessor,
        svd_preprocessor=svd_preprocessor,
        svd_preprocessor_path=svd_preprocessor_path,
    )
    speaker_manifest = load_speaker_manifest(data_root, speaker_id)
    speaker_dir = data_root / speaker_id
    graphs: list[GraphSample] = []

    for item in speaker_manifest.get("files", []):
        record_path = speaker_dir / item["file"]
        with record_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        graphs.append(_record_to_graph(record, svd_preprocessor=resolved_preprocessor))

    return graphs


def build_utterance_graphs(
    data_root: str | Path,
    speaker_ids: Iterable[str] | None = None,
    use_svd_preprocessor: bool = False,
    svd_preprocessor: Optional[SVDNodeFeaturePreprocessor] = None,
    svd_preprocessor_path: str | Path | None = None,
) -> list[GraphSample]:
    data_root = Path(data_root)
    resolved_preprocessor = _resolve_preprocessor(
        use_svd_preprocessor=use_svd_preprocessor,
        svd_preprocessor=svd_preprocessor,
        svd_preprocessor_path=svd_preprocessor_path,
    )
    if speaker_ids is None:
        manifest_path = data_root / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as file:
            global_manifest = json.load(file)
        speaker_ids = [entry["speaker_id"] for entry in global_manifest.get("speakers", [])]

    graphs = []
    for speaker_id in speaker_ids:
        graphs.extend(
            load_speaker_graphs(
                data_root=data_root,
                speaker_id=speaker_id,
                use_svd_preprocessor=resolved_preprocessor is not None,
                svd_preprocessor=resolved_preprocessor,
                svd_preprocessor_path=None,
            )
        )
    return graphs
