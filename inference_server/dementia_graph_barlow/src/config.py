from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainingConfig:
    hidden_dim: int = 64
    projection_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    lambda_offdiag: float = 0.005
    mask_ratio: float = 0.2
    noise_std: float = 0.01
    seed: int = 42


@dataclass
class DataConfig:
    timestamp_col: str = "timestamp"
    household_col: str = "household_id"
    sensor_col: str = "sensor_id"
    sensor_value_col: Optional[str] = None
    sensor_value_mode: str = "scalar"
    temporal_window: str = "day"
    transition_max_gap_minutes: int = 90


@dataclass
class AlertConfig:
    target_alert_rate: float = 0.07
    calibration_ratio: float = 0.5
    anomaly_metric: str = "cosine"


@dataclass
class PipelineConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)


def _merge_dict(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None) -> PipelineConfig:
    defaults = {
        "training": TrainingConfig().__dict__,
        "data": DataConfig().__dict__,
        "alert": AlertConfig().__dict__,
    }

    if config_path is None:
        merged = defaults
    else:
        with open(config_path, "r", encoding="utf-8") as file:
            user_cfg = yaml.safe_load(file) or {}
        merged = _merge_dict(defaults, user_cfg)

    return PipelineConfig(
        training=TrainingConfig(**merged["training"]),
        data=DataConfig(**merged["data"]),
        alert=AlertConfig(**merged["alert"]),
    )
