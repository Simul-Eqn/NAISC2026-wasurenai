from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


@dataclass
class DailyGraph:
    household_id: str
    period_start: date
    x: torch.Tensor
    adj: torch.Tensor
    sensor_names: List[str]


def load_sensor_events(
    csv_path: str,
    timestamp_col: str,
    household_col: str,
    sensor_col: str,
    sensor_value_col: Optional[str] = None,
    temporal_window: str = "day",
) -> pd.DataFrame:
    events = pd.read_csv(csv_path)
    required = {timestamp_col, household_col, sensor_col}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    events = events.copy()
    events[timestamp_col] = pd.to_datetime(events[timestamp_col], utc=True, errors="coerce")
    events = events.dropna(subset=[timestamp_col])
    events[household_col] = events[household_col].astype(str)
    events[sensor_col] = events[sensor_col].astype(str)

    if temporal_window == "day":
        events["_period_start"] = events[timestamp_col].dt.date
    elif temporal_window == "week":
        ts_naive = events[timestamp_col].dt.tz_localize(None)
        events["_period_start"] = ts_naive.dt.to_period("W-SUN").dt.start_time.dt.date
    else:
        raise ValueError(f"Unsupported temporal_window: {temporal_window}. Use 'day' or 'week'.")

    return events.sort_values([household_col, timestamp_col]).reset_index(drop=True)


def build_sensor_index(events: pd.DataFrame, sensor_col: str) -> Dict[str, int]:
    sensors = sorted(events[sensor_col].unique().tolist())
    return {sensor: idx for idx, sensor in enumerate(sensors)}


def _build_node_features(
    day_events: pd.DataFrame,
    sensor_idx: Dict[str, int],
    sensor_col: str,
    timestamp_col: str,
    sensor_value_col: Optional[str] = None,
    sensor_value_mode: str = "scalar",
    sensor_value_dim: Optional[int] = None,
) -> np.ndarray:
    n_nodes = len(sensor_idx)
    has_value = sensor_value_col is not None and sensor_value_col in day_events.columns

    if sensor_value_mode not in {"scalar", "vector"}:
        raise ValueError(f"Unsupported sensor_value_mode: {sensor_value_mode}. Use 'scalar' or 'vector'.")

    if has_value and sensor_value_mode == "vector" and sensor_value_dim is not None and sensor_value_dim > 0:
        feature_dim = 3 + 2 * sensor_value_dim
    else:
        feature_dim = 5 if has_value else 3

    features = np.zeros((n_nodes, feature_dim), dtype=np.float32)

    for sensor_name, group in day_events.groupby(sensor_col):
        idx = sensor_idx[sensor_name]
        timestamps = group[timestamp_col]
        hours = timestamps.dt.hour.to_numpy(dtype=np.float32) + timestamps.dt.minute.to_numpy(dtype=np.float32) / 60.0

        features[idx, 0] = float(len(group))
        features[idx, 1] = float(hours.mean() / 24.0)
        features[idx, 2] = float(hours.std() / 24.0) if len(hours) > 1 else 0.0

        if has_value:
            if sensor_value_mode == "vector" and sensor_value_dim is not None and sensor_value_dim > 0:
                vectors = _extract_fixed_vectors(group[sensor_value_col], sensor_value_dim)
                if vectors.size > 0:
                    mean_vec = vectors.mean(axis=0)
                    std_vec = vectors.std(axis=0) if vectors.shape[0] > 1 else np.zeros(sensor_value_dim, dtype=np.float32)
                    features[idx, 3 : 3 + sensor_value_dim] = mean_vec
                    features[idx, 3 + sensor_value_dim : 3 + 2 * sensor_value_dim] = std_vec
            else:
                values = group[sensor_value_col].apply(_sensor_value_to_scalar).dropna().to_numpy(dtype=np.float32)
                if values.size > 0:
                    features[idx, 3] = float(values.mean())
                    features[idx, 4] = float(values.std()) if values.size > 1 else 0.0

    max_count = features[:, 0].max()
    if max_count > 0:
        features[:, 0] /= max_count

    if has_value:
        if sensor_value_mode == "vector" and sensor_value_dim is not None and sensor_value_dim > 0:
            mean_block = features[:, 3 : 3 + sensor_value_dim]
            std_block = features[:, 3 + sensor_value_dim : 3 + 2 * sensor_value_dim]

            mean_scale = np.max(np.abs(mean_block), axis=0)
            mean_scale[mean_scale == 0] = 1.0
            features[:, 3 : 3 + sensor_value_dim] = mean_block / mean_scale

            std_scale = np.max(std_block, axis=0)
            std_scale[std_scale == 0] = 1.0
            features[:, 3 + sensor_value_dim : 3 + 2 * sensor_value_dim] = std_block / std_scale
        else:
            max_abs_mean = np.abs(features[:, 3]).max()
            if max_abs_mean > 0:
                features[:, 3] /= max_abs_mean
            max_std = features[:, 4].max()
            if max_std > 0:
                features[:, 4] /= max_std

    return features


def _sensor_value_to_scalar(value: object) -> float:
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32).reshape(-1)
        if arr.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(arr * arr)))

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return np.nan
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(arr * arr)))

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return np.nan

        try:
            return float(text)
        except ValueError:
            pass

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return np.nan

            if not isinstance(parsed, list) or len(parsed) == 0:
                return np.nan

            arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return np.nan
            return float(np.sqrt(np.mean(arr * arr)))

    return np.nan


def _sensor_value_to_vector(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32).reshape(-1)
        return arr if arr.size > 0 else None

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr if arr.size > 0 else None

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, list) or len(parsed) == 0:
                return None
            arr = np.asarray(parsed, dtype=np.float32).reshape(-1)
            return arr if arr.size > 0 else None

    return None


def _infer_sensor_value_vector_dim(events: pd.DataFrame, sensor_value_col: str) -> Optional[int]:
    if sensor_value_col not in events.columns:
        return None

    for value in events[sensor_value_col].tolist():
        vector = _sensor_value_to_vector(value)
        if vector is not None and vector.size > 0:
            return int(vector.size)
    return None


def _extract_fixed_vectors(values: pd.Series, expected_dim: int) -> np.ndarray:
    vectors = []
    for value in values.tolist():
        vector = _sensor_value_to_vector(value)
        if vector is None:
            continue
        if vector.size != expected_dim:
            continue
        vectors.append(vector)

    if not vectors:
        return np.empty((0, expected_dim), dtype=np.float32)

    return np.vstack(vectors).astype(np.float32)


def _build_transition_adj(
    day_events: pd.DataFrame,
    sensor_idx: Dict[str, int],
    sensor_col: str,
    timestamp_col: str,
    transition_max_gap_minutes: int,
) -> np.ndarray:
    n_nodes = len(sensor_idx)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    ordered = day_events.sort_values(timestamp_col)
    sensors = ordered[sensor_col].tolist()
    timestamps = ordered[timestamp_col].tolist()

    for i in range(len(ordered) - 1):
        dt = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60.0
        if dt > transition_max_gap_minutes:
            continue

        src = sensor_idx[sensors[i]]
        dst = sensor_idx[sensors[i + 1]]
        adj[src, dst] += 1.0

    np.fill_diagonal(adj, np.diag(adj) + 1.0)
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return adj / row_sum


def build_daily_graphs(
    events: pd.DataFrame,
    household_col: str,
    sensor_col: str,
    timestamp_col: str,
    transition_max_gap_minutes: int,
    sensor_value_col: Optional[str] = None,
    sensor_value_mode: str = "scalar",
) -> Dict[str, List[DailyGraph]]:
    sensor_idx = build_sensor_index(events, sensor_col)
    sensor_value_dim = None
    if sensor_value_col is not None and sensor_value_col in events.columns and sensor_value_mode == "vector":
        sensor_value_dim = _infer_sensor_value_vector_dim(events, sensor_value_col)

    sensor_names = [None] * len(sensor_idx)
    for name, idx in sensor_idx.items():
        sensor_names[idx] = name

    output: Dict[str, List[DailyGraph]] = {}
    for household_id, hh_events in events.groupby(household_col):
        household_graphs: List[DailyGraph] = []
        for period_start, day_events in hh_events.groupby("_period_start"):
            x_np = _build_node_features(
                day_events,
                sensor_idx,
                sensor_col,
                timestamp_col,
                sensor_value_col=sensor_value_col,
                sensor_value_mode=sensor_value_mode,
                sensor_value_dim=sensor_value_dim,
            )
            adj_np = _build_transition_adj(
                day_events,
                sensor_idx,
                sensor_col,
                timestamp_col,
                transition_max_gap_minutes,
            )
            household_graphs.append(
                DailyGraph(
                    household_id=str(household_id),
                    period_start=period_start,
                    x=torch.from_numpy(x_np),
                    adj=torch.from_numpy(adj_np),
                    sensor_names=sensor_names,
                )
            )

        household_graphs.sort(key=lambda g: g.period_start)
        output[str(household_id)] = household_graphs

    return output


def aggregation_feature_mask(graph: DailyGraph, mask_ratio: float, noise_std: float) -> torch.Tensor:
    x = graph.x
    adj = graph.adj
    n_nodes = x.size(0)

    n_mask = max(1, int(round(mask_ratio * n_nodes)))
    perm = torch.randperm(n_nodes)
    mask_idx = perm[:n_mask]

    eye = torch.eye(n_nodes, device=x.device, dtype=x.dtype)
    adj_hat = adj + eye
    degree = adj_hat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    agg = torch.matmul(adj_hat / degree, x)

    augmented = x.clone()
    augmented[mask_idx] = agg[mask_idx]
    if noise_std > 0:
        augmented[mask_idx] = augmented[mask_idx] + noise_std * torch.randn_like(augmented[mask_idx])
    return augmented


def sensor_count_by_household(events: pd.DataFrame, household_col: str, sensor_col: str) -> pd.DataFrame:
    counts = (
        events.groupby(household_col)[sensor_col]
        .nunique()
        .reset_index()
        .rename(columns={household_col: "household_id", sensor_col: "n_sensors"})
    )
    counts["single_sensor_only"] = counts["n_sensors"] <= 1
    return counts.sort_values("household_id").reset_index(drop=True)
