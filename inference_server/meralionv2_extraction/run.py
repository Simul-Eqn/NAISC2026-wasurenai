from __future__ import annotations

import csv
import gc
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

REPO_ID = "MERaLiON/MERaLiON-SpeechEncoder-2"
DATASET_NAME = "CAiRE/ASCEND"
SPLIT = "validation"
BATCH_SIZE = 40
START_DATETIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
OUTPUT_CSV = Path("inference_server/meralionv2_extraction/outputs/ascend_graph_events.csv")
MEMORY_LOG_EVERY = 10
ENABLE_LOGGING = True


def id_sort_key(sample_id: str) -> tuple[int, str]:
    sample_id = str(sample_id)
    if sample_id.isdigit():
        return int(sample_id), sample_id
    return 10**12, sample_id


def build_timestamp_map(dataset: Dataset, start_datetime: datetime) -> Dict[int, datetime]:
    ids = dataset["id"]
    durations = dataset["duration"]
    sessions = dataset["session_id"]

    order = sorted(range(len(dataset)), key=lambda idx: id_sort_key(ids[idx]))
    timestamp_map: Dict[int, datetime] = {}

    cumulative_seconds = 0.0
    for idx in order:
        session_days = int(sessions[idx])
        timestamp_map[idx] = start_datetime + timedelta(days=session_days, seconds=cumulative_seconds)
        cumulative_seconds += float(durations[idx])

    return timestamp_map


def anomaly_decider(sample: dict) -> bool: #, sensor_value: float) -> bool:
    duration = float(sample.get("duration", 0.0))
    transcription = str(sample.get("transcription", "")).strip()
    topic = str(sample.get("topic", "")).strip()

    too_short_or_long = duration < 0.4 or duration > 20.0
    sparse_text = len(transcription) < 4
    #extreme_sensor_value = abs(sensor_value) > 50.0
    missing_topic = len(topic) == 0

    return too_short_or_long or sparse_text or missing_topic #or extreme_sensor_value or missing_topic


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

    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    timestamp_map = build_timestamp_map(dataset, START_DATETIME)

    ordered_indices = sorted(range(len(dataset)), key=lambda idx: id_sort_key(dataset[idx]["id"]))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "timestamp",
        "household_id",
        "sensor_id",
        "sensor_raw_values",
        "transcription",
        "is_anomaly",
        "duration",
        "session_id",
        "language",
    ]

    model.eval()
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

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
                #pooled = mean_pool_last_hidden(output)
                #sensor_values = torch.norm(pooled, dim=1)

                for local_idx, sample in enumerate(batch_samples):
                    original_idx = batch_indices[local_idx]
                    sensor_feature_sequence = extracted_cpu[local_idx].tolist()
                    #sensor_value = float(sensor_values[local_idx].detach().cpu().item())

                    row = {
                        "id": str(sample["id"]),
                        "timestamp": timestamp_map[original_idx].isoformat(),
                        "household_id": str(sample["original_speaker_id"]),
                        "sensor_id": "AUDIO", #str(sample["topic"]),
                        #"sensor_value": sensor_value,
                        "sensor_raw_values": json.dumps(sensor_feature_sequence, ensure_ascii=False),
                        "transcription": str(sample.get("transcription", "")),
                        #"is_anomaly": anomaly_decider(sample, sensor_value),
                        "is_anomaly": anomaly_decider(sample),
                        "duration": float(sample.get("duration", 0.0)),
                        "session_id": int(sample.get("session_id", 0)),
                        "language": str(sample.get("language", "")),
                    }
                    writer.writerow(row)

                del output
                del extracted_cpu
                del inputs
                del audio_arrays
                del batch_ds
                del batch_samples
                del sampling_rates
                del row
                gc.collect()

                if device == "cuda":
                    torch.cuda.synchronize()

                batch_elapsed = time.perf_counter() - batch_start_time
                if batch_id % MEMORY_LOG_EVERY == 0:
                    log_batch_speed(f"batch_{batch_id}", len(batch_indices), batch_elapsed)

                if device == "cuda" and batch_id % MEMORY_LOG_EVERY == 0:
                    log_gpu_memory(f"batch_{batch_id}_end")

    print(f"Finished saving rows incrementally to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
