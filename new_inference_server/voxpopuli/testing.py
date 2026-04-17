from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path
from typing import Any


_serialization_verified = False


def normalize_accent_value(record: dict[str, Any]) -> dict[str, Any]:
    record.pop("has_accent", None)
    return record


def flag_error(path: Path, message: str) -> None:
    flag_path = path.with_suffix(path.suffix + ".flag")
    with flag_path.open("w", encoding="utf-8") as file:
        file.write(message)
    print(f"FLAGGED: {path}")
    print(message)


def write_compact_json_atomic(path: Path, payload: Any) -> None:
    global _serialization_verified

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    json.dump(payload, buffer, ensure_ascii=False, separators=(",", ":"))

    if not _serialization_verified:
        json.loads(buffer.getvalue())
        _serialization_verified = True

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        file.write(buffer.getvalue())
    tmp_path.replace(path)


def read_record(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = ast.literal_eval(text)

    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict-like record, got {type(loaded).__name__}")
    return loaded


def fix_record_file(path: Path) -> bool:
    try:
        record = read_record(path)

        updated = normalize_accent_value(record)
        write_compact_json_atomic(path, updated)
        return True
    except Exception as exc:
        flag_error(path, f"{type(exc).__name__}: {exc}")
        return False


def iter_json_files(root: Path):
    for path in root.rglob("*.json"):
        if path.name == "manifest.json":
            continue
        yield path


def is_speaker_folder(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    for child in folder.iterdir():
        if child.is_file() and child.suffix == ".json" and child.name != "manifest.json":
            return True
    return False


def rebuild_manifests(data_root: Path) -> None:
    global_manifest: dict[str, Any] = {"speakers": []}

    speaker_dirs = [Path("new_inference_server/voxpopuli/data/1185")] #sorted([folder for folder in data_root.iterdir() if is_speaker_folder(folder)], key=lambda p: p.name)
    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name
        entries: list[dict[str, Any]] = []

        speaker_files = sorted(
            [path for path in speaker_dir.iterdir() if path.is_file() and path.suffix == ".json" and path.name != "manifest.json"],
            key=lambda p: p.name,
        )

        for file_path in speaker_files:
            try:
                record = read_record(file_path)

                entries.append(
                    {
                        "file": file_path.name,
                        "audio_id": record.get("audio_id"),
                        "speaker_id": record.get("speaker_id", speaker_id),
                        "accent_label": record.get("accent_label"),
                        "language": record.get("language"),
                    }
                )
            except Exception as exc:
                flag_error(file_path, f"{type(exc).__name__}: {exc}")
                continue

        speaker_manifest_path = speaker_dir / "manifest.json"
        write_compact_json_atomic(
            speaker_manifest_path,
            {
                "speaker_id": speaker_id,
                "count": len(entries),
                "files": entries,
            },
        )

        global_manifest["speakers"].append(
            {
                "speaker_id": speaker_id,
                "count": len(entries),
                "manifest": str(Path(speaker_id) / "manifest.json"),
            }
        )

    global_manifest["speakers"].sort(key=lambda item: item["speaker_id"])
    global_manifest_path = data_root / "manifest.json"
    write_compact_json_atomic(global_manifest_path, global_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize VoxPopuli accent labels")
    parser.add_argument(
        "--data_root",
        type=str,
        default="new_inference_server/voxpopuli/data",
        help="Root folder containing speaker subfolders",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)


    rebuild_manifests(data_root)
    print(f"Rebuilt speaker manifests and global manifest under {data_root}")


if __name__ == "__main__":
    main()
