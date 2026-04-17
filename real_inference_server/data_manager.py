"""Data manager for storing and retrieving mel spectrograms per user."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import DATA_DIR, METADATA_DIR


class DataManager:
    """Manages storage and retrieval of user data."""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.metadata_dir = METADATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _get_user_dir(self, chat_id: int) -> Path:
        """Get the directory for a specific user."""
        user_dir = self.data_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_metadata_path(self, chat_id: int) -> Path:
        """Get the metadata file path for a user."""
        return self.metadata_dir / f"{chat_id}_metadata.json"

    def save_mel_spectrogram(self, chat_id: int, mel_spec: np.ndarray) -> str:
        """Save a mel spectrogram for a user.

        Returns:
            The filename of the saved spectrogram.
        """
        user_dir = self._get_user_dir(chat_id)
        # Count existing samples to create a sequential filename
        existing_files = list(user_dir.glob("mel_spec_*.npy"))
        sample_index = len(existing_files)
        filename = f"mel_spec_{sample_index:05d}.npy"
        filepath = user_dir / filename

        np.save(filepath, mel_spec.astype(np.float32))
        return filename

    def load_user_mel_spectrograms(self, chat_id: int) -> list[tuple[str, np.ndarray]]:
        """Load all mel spectrograms for a user.

        Returns:
            List of tuples (filename, mel_spectrogram).
        """
        user_dir = self._get_user_dir(chat_id)
        files = sorted(user_dir.glob("mel_spec_*.npy"))
        spectrograms = []
        for filepath in files:
            spec = np.load(filepath)
            spectrograms.append((filepath.name, spec))
        return spectrograms

    def get_user_metadata(self, chat_id: int) -> dict[str, Any]:
        """Get metadata for a user."""
        metadata_path = self._get_metadata_path(chat_id)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "chat_id": chat_id,
            "num_samples": 0,
            "is_trained": False,
            "threshold": None,
            "mean_score": None,
            "std_score": None,
        }

    def save_user_metadata(self, chat_id: int, metadata: dict[str, Any]) -> None:
        """Save metadata for a user."""
        metadata_path = self._get_metadata_path(chat_id)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def update_user_metadata(self, chat_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        """Update metadata for a user."""
        metadata = self.get_user_metadata(chat_id)
        metadata.update(updates)
        self.save_user_metadata(chat_id, metadata)
        return metadata

    def get_num_samples(self, chat_id: int) -> int:
        """Get the number of samples collected for a user."""
        user_dir = self._get_user_dir(chat_id)
        return len(list(user_dir.glob("mel_spec_*.npy")))

    def clear_user_data(self, chat_id: int) -> None:
        """Clear all data for a user (for testing/reset)."""
        user_dir = self._get_user_dir(chat_id)
        for filepath in user_dir.glob("mel_spec_*.npy"):
            filepath.unlink()
        metadata_path = self._get_metadata_path(chat_id)
        if metadata_path.exists():
            metadata_path.unlink()
