from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_global_manifest(data_root: str | Path) -> dict[str, Any]:
    data_root = Path(data_root)
    manifest_path = data_root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_speakers(data_root: str | Path) -> list[str]:
    manifest = load_global_manifest(data_root)
    return [entry["speaker_id"] for entry in manifest.get("speakers", [])]


def load_speaker_manifest(data_root: str | Path, speaker_id: str) -> dict[str, Any]:
    data_root = Path(data_root)
    manifest_path = data_root / speaker_id / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def iter_speaker_records(data_root: str | Path, speaker_id: str):
    data_root = Path(data_root)
    manifest = load_speaker_manifest(data_root, speaker_id)
    speaker_dir = data_root / speaker_id

    for item in manifest.get("files", []):
        record_path = speaker_dir / item["file"]
        with record_path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        yield record


def load_speaker_embeddings(data_root: str | Path, speaker_id: str) -> list[dict[str, Any]]:
    records = []
    for record in iter_speaker_records(data_root, speaker_id):
        records.append(
            {
                "audio_id": record.get("audio_id"),
                "speaker_id": record.get("speaker_id"),
                "has_accent": record.get("has_accent"),
                "accent_label": record.get("accent_label"),
                "language": record.get("language"),
                "audio_embedding": record.get("audio_embedding"),
            }
        )
    return records
