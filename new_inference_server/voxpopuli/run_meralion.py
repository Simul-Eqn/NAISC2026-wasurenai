from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

REPO_ID = "MERaLiON/MERaLiON-SpeechEncoder-2"
DATASET_NAME = "facebook/voxpopuli"
DATASET_CONFIG = "en"
SPLIT = "train"
BATCH_SIZE = 20
OUTPUT_ROOT = Path("new_inference_server/voxpopuli/data")
MEMORY_LOG_EVERY = 10
ENABLE_LOGGING = True


def id_sort_key(sample_id: str) -> tuple[int, str]:
    sample_id = str(sample_id)
    if sample_id.isdigit():
        return int(sample_id), sample_id
    return 10**12, sample_id


def _pick_first_existing(sample: dict, candidates: list[str], default: Any = None) -> Any:
    for key in candidates:
        if key in sample and sample[key] is not None:
            return sample[key]
    return default


def extract_speaker_id(sample: dict) -> str:
    speaker_val = _pick_first_existing(
        sample,
        candidates=["speaker_id", "speaker", "original_speaker_id", "client_id", "speakerId"],
        default="unknown_speaker",
    )
    return str(speaker_val)


def extract_audio_id(sample: dict) -> str:
    audio_id = _pick_first_existing(sample, candidates=["audio_id", "id", "utterance_id"], default=None)
    if audio_id is not None:
        return str(audio_id)

    audio_meta = sample.get("audio", {})
    if isinstance(audio_meta, dict):
        path = audio_meta.get("path")
        if path:
            return str(Path(path).stem)
    return "unknown_audio_id"


def extract_sort_id(sample: dict) -> str:
    sortable = _pick_first_existing(sample, candidates=["audio_id", "id", "utterance_id"], default=None)
    if sortable is not None:
        return str(sortable)
    return extract_audio_id(sample)


def extract_accent_flag(sample: dict) -> tuple[bool | None, str | None]:
    accent_val = _pick_first_existing(
        sample,
        candidates=["accent", "accents", "speaker_accent", "region", "dialect"],
        default=None,
    )

    if accent_val is None:
        return None, None

    accent_text = str(accent_val).strip()
    if accent_text == "" or accent_text.lower() in {"none", "null", "nan"}:
        return False, None
    return True, accent_text


def log_gpu_memory(tag: str) -> None:
    if not ENABLE_LOGGING:
        return
    if not torch.cuda.is_available():
        return
    allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    reserved_mb = torch.cuda.memory_reserved() / (1024**2)
    max_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(
        f"[GPU:{tag}] allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB max_allocated={max_allocated_mb:.1f}MB"
    )


def log_batch_speed(tag: str, batch_size: int, elapsed_s: float) -> None:
    if not ENABLE_LOGGING:
        return
    samples_per_s = batch_size / elapsed_s if elapsed_s > 0 else float("inf")
    print(f"[SPEED:{tag}] batch_time={elapsed_s:.3f}s throughput={samples_per_s:.2f} samples/s")


'''def mean_pool_last_hidden(output: object) -> torch.Tensor:
    features = output.extract_features
    return features.mean(dim=1)'''


def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = AutoModel.from_pretrained(
        REPO_ID,
        trust_remote_code=True,
        dtype="auto",
    ).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        REPO_ID,
        trust_remote_code=True,
        dtype="auto",
    )

    dataset: Dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=SPLIT)

    ordered_indices = sorted(range(len(dataset)), key=lambda idx: id_sort_key(extract_sort_id(dataset[idx])))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    speaker_file_counter: dict[str, int] = {}
    speaker_manifest_entries: dict[str, list[dict[str, Any]]] = {}

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ordered_indices), BATCH_SIZE):
            if ENABLE_LOGGING:
                print(start)
            batch_id = start // BATCH_SIZE
            if device == "cuda" and batch_id % MEMORY_LOG_EVERY == 0:
                log_gpu_memory(f"batch_{batch_id}_start")

            batch_start_time = time.perf_counter()

            batch_indices = ordered_indices[start : start + BATCH_SIZE]
            batch_ds = dataset.select(batch_indices)
            batch_samples = [batch_ds[i] for i in range(len(batch_ds))]

            audio_arrays = [sample["audio"]["array"] for sample in batch_samples]
            sampling_rates = [int(sample["audio"]["sampling_rate"]) for sample in batch_samples]
            if len(set(sampling_rates)) != 1:
                raise ValueError(f"Mixed sampling rates in batch: {sorted(set(sampling_rates))}")

            inputs = feature_extractor(
                audio_arrays,
                sampling_rate=sampling_rates[0],
                return_attention_mask=True,
                return_tensors="pt",
                do_normalize=False,
            ).to(device)

            output = model(
                input_values=inputs["input_values"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=False,
            )

            extracted_cpu = output.extract_features.detach().cpu()

            for local_idx, sample in enumerate(batch_samples):
                sensor_feature_sequence = extracted_cpu[local_idx].tolist()
                speaker_id = extract_speaker_id(sample)
                audio_id = extract_audio_id(sample)
                has_accent, accent_label = extract_accent_flag(sample)

                speaker_dir = OUTPUT_ROOT / speaker_id
                speaker_dir.mkdir(parents=True, exist_ok=True)

                speaker_file_counter[speaker_id] = speaker_file_counter.get(speaker_id, 0) + 1
                file_name = f"file_{speaker_file_counter[speaker_id]:06d}.json"
                output_file = speaker_dir / file_name

                payload = {
                    "audio_id": audio_id,
                    "speaker_id": speaker_id,
                    "has_accent": has_accent,
                    "accent_label": accent_label,
                    "language": str(sample.get("language", "")),
                    "audio_embedding": sensor_feature_sequence,
                }

                with output_file.open("w", encoding="utf-8") as file:
                    json.dump(payload, file, ensure_ascii=False)

                if speaker_id not in speaker_manifest_entries:
                    speaker_manifest_entries[speaker_id] = []
                speaker_manifest_entries[speaker_id].append(
                    {
                        "file": output_file.name,
                        "audio_id": audio_id,
                        "speaker_id": speaker_id,
                        "has_accent": has_accent,
                        "accent_label": accent_label,
                        "language": str(sample.get("language", "")),
                    }
                )

            del output
            del extracted_cpu
            del inputs
            del audio_arrays
            del batch_ds
            del batch_samples
            del sampling_rates
            gc.collect()

            if device == "cuda":
                torch.cuda.synchronize()

            batch_elapsed = time.perf_counter() - batch_start_time
            if batch_id % MEMORY_LOG_EVERY == 0:
                log_batch_speed(f"batch_{batch_id}", len(batch_indices), batch_elapsed)

            if device == "cuda" and batch_id % MEMORY_LOG_EVERY == 0:
                log_gpu_memory(f"batch_{batch_id}_end")

    global_manifest = {"speakers": []}
    for speaker_id, entries in speaker_manifest_entries.items():
        speaker_dir = OUTPUT_ROOT / speaker_id
        speaker_manifest_path = speaker_dir / "manifest.json"
        speaker_manifest = {
            "speaker_id": speaker_id,
            "count": len(entries),
            "files": entries,
        }
        with speaker_manifest_path.open("w", encoding="utf-8") as file:
            json.dump(speaker_manifest, file, ensure_ascii=False, indent=2)

        global_manifest["speakers"].append(
            {
                "speaker_id": speaker_id,
                "count": len(entries),
                "manifest": str(speaker_manifest_path.relative_to(OUTPUT_ROOT)),
            }
        )

    global_manifest["speakers"].sort(key=lambda item: item["speaker_id"])
    global_manifest_path = OUTPUT_ROOT / "manifest.json"
    with global_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(global_manifest, file, ensure_ascii=False, indent=2)

    print(f"Finished saving per-sample JSON files under {OUTPUT_ROOT}")
    if ENABLE_LOGGING:
        print(f"Wrote per-speaker manifests and global manifest at {global_manifest_path}")


if __name__ == "__main__":
    run()
