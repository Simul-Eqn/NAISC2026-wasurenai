from __future__ import annotations

import argparse
import ast
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = MODULE_DIR / "data"
DEFAULT_OUTPUT_PATH = MODULE_DIR / "svd_preprocessor.npz"


@dataclass
class SVDNodeFeaturePreprocessor:
    n_components: int
    input_dim: int
    components: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    pad_atol: float = 1e-6

    @classmethod
    def fit_from_data_root(
        cls,
        data_root: str | Path,
        n_components: int,
        pad_atol: float = 1e-6,
        random_state: int = 42,
    ) -> "SVDNodeFeaturePreprocessor":
        data_root = Path(data_root)
        matrices: list[sparse.csr_matrix] = []
        input_dim: Optional[int] = None

        for embedding in iter_all_embeddings(data_root, pad_atol=pad_atol):
            if embedding.ndim != 2:
                raise ValueError(f"Expected 2D embedding matrix, got shape {embedding.shape}")
            if input_dim is None:
                input_dim = int(embedding.shape[1])
            elif embedding.shape[1] != input_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension: expected {input_dim}, got {embedding.shape[1]}"
                )
            matrices.append(sparse.csr_matrix(embedding, dtype=np.float32))

        if not matrices:
            raise ValueError(f"No embeddings found under {data_root}")
        if input_dim is None:
            raise ValueError("Unable to infer input_dim")

        matrix = sparse.vstack(matrices, format="csr")
        if matrix.shape[0] < 2:
            raise ValueError(f"Need at least 2 rows to fit SVD, got {matrix.shape[0]}")

        max_components = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
        if max_components < 1:
            raise ValueError(
                f"Cannot fit SVD with n_components={n_components} on matrix shape {matrix.shape}"
            )

        svd = TruncatedSVD(n_components=max_components, random_state=random_state)
        svd.fit(matrix)

        return cls(
            n_components=int(max_components),
            input_dim=int(input_dim),
            components=svd.components_.astype(np.float32),
            singular_values=svd.singular_values_.astype(np.float32),
            explained_variance=svd.explained_variance_.astype(np.float32),
            explained_variance_ratio=svd.explained_variance_ratio_.astype(np.float32),
            pad_atol=float(pad_atol),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SVDNodeFeaturePreprocessor":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            return cls(
                n_components=int(data["n_components"]),
                input_dim=int(data["input_dim"]),
                components=data["components"].astype(np.float32),
                singular_values=data["singular_values"].astype(np.float32),
                explained_variance=data["explained_variance"].astype(np.float32),
                explained_variance_ratio=data["explained_variance_ratio"].astype(np.float32),
                pad_atol=float(data["pad_atol"]) if "pad_atol" in data else 1e-6,
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            n_components=np.array(self.n_components, dtype=np.int64),
            input_dim=np.array(self.input_dim, dtype=np.int64),
            components=self.components.astype(np.float32),
            singular_values=self.singular_values.astype(np.float32),
            explained_variance=self.explained_variance.astype(np.float32),
            explained_variance_ratio=self.explained_variance_ratio.astype(np.float32),
            pad_atol=np.array(self.pad_atol, dtype=np.float32),
        )

    def transform(self, node_features: np.ndarray) -> np.ndarray:
        arr = np.asarray(node_features, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"node_features must be 2D, got shape {arr.shape}")
        if arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {arr.shape[1]}")
        return (arr @ self.components.T).astype(np.float32)


def _read_record(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = ast.literal_eval(text)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict-like record, got {type(loaded).__name__}")
    return loaded


def _trim_trailing_padding(audio_embedding: Any, pad_atol: float = 1e-6) -> np.ndarray:
    arr = np.asarray(audio_embedding, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"audio_embedding must be 1D or 2D, got shape {arr.shape}")
    row_norms = np.linalg.norm(arr, axis=1)
    valid_rows = np.flatnonzero(row_norms > pad_atol)
    if valid_rows.size == 0:
        return arr[:1]
    return arr[: int(valid_rows[-1]) + 1]


def iter_record_paths(data_root: Path) -> Iterator[Path]:
    for speaker_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        for file_path in sorted(speaker_dir.glob("*.json"), key=lambda p: p.name):
            if file_path.name == "manifest.json":
                continue
            yield file_path


def iter_all_embeddings(data_root: str | Path, pad_atol: float = 1e-6) -> Iterator[np.ndarray]:
    data_root = Path(data_root)
    for record_path in iter_record_paths(data_root):
        record = _read_record(record_path)
        emb = _trim_trailing_padding(record.get("audio_embedding", []), pad_atol=pad_atol)
        if emb.size == 0:
            continue
        yield emb


def fit_and_save(
    data_root: str | Path,
    output_path: str | Path,
    n_components: int,
    pad_atol: float = 1e-6,
    random_state: int = 42,
) -> SVDNodeFeaturePreprocessor:
    preprocessor = SVDNodeFeaturePreprocessor.fit_from_data_root(
        data_root=data_root,
        n_components=n_components,
        pad_atol=pad_atol,
        random_state=random_state,
    )
    preprocessor.save(output_path)
    return preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an SVD preprocessor for VoxPopuli node embeddings")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output_path", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--n_components", type=int, default=256)
    parser.add_argument("--pad_atol", type=float, default=1e-6)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    preprocessor = fit_and_save(
        data_root=args.data_root,
        output_path=args.output_path,
        n_components=args.n_components,
        pad_atol=args.pad_atol,
        random_state=args.random_state,
    )
    print(
        f"Saved SVD preprocessor to {args.output_path} with n_components={preprocessor.n_components}, input_dim={preprocessor.input_dim}"
    )
    # Saved SVD preprocessor to new_inference_server/voxpopuli/svd_preprocessor.npz with n_components=256, input_dim=19456 


if __name__ == "__main__":
    main()
