from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from .anomaly import apply_alerting, compute_daily_drift_scores, fit_personalized_thresholds
from .config import PipelineConfig
from .data import build_daily_graphs, load_sensor_events, sensor_count_by_household
from .train import encode_graphs, flatten_graphs, save_checkpoint, train_graph_barlow


def run_pipeline(events_csv: str, config: PipelineConfig, output_dir: str) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    events = load_sensor_events(
        csv_path=events_csv,
        timestamp_col=config.data.timestamp_col,
        household_col=config.data.household_col,
        sensor_col=config.data.sensor_col,
        sensor_value_col=config.data.sensor_value_col,
        temporal_window=config.data.temporal_window,
    )
    sensor_profile_df = sensor_count_by_household(
        events,
        household_col=config.data.household_col,
        sensor_col=config.data.sensor_col,
    )
    graphs_by_household = build_daily_graphs(
        events=events,
        household_col=config.data.household_col,
        sensor_col=config.data.sensor_col,
        timestamp_col=config.data.timestamp_col,
        transition_max_gap_minutes=config.data.transition_max_gap_minutes,
        sensor_value_col=config.data.sensor_value_col,
        sensor_value_mode=config.data.sensor_value_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_graph_barlow(flatten_graphs(graphs_by_household), config.training, device)

    encoded = encode_graphs(model, graphs_by_household, device)
    scores_df = compute_daily_drift_scores(encoded, metric=config.alert.anomaly_metric)
    thresholds_df = fit_personalized_thresholds(
        scores_df,
        target_alert_rate=config.alert.target_alert_rate,
        calibration_ratio=config.alert.calibration_ratio,
    )
    alerts_df = apply_alerting(scores_df, thresholds_df)

    ckpt_path = output_path / "graph_barlow_twins.pt"
    save_checkpoint(model, config.training, ckpt_path)

    scores_path = output_path / "daily_scores.csv"
    thresholds_path = output_path / "personalized_thresholds.csv"
    alerts_path = output_path / "alerts.csv"
    sensor_profile_path = output_path / "household_sensor_profile.csv"

    period_key = "period_start" if "period_start" in scores_df.columns else "day"
    scores_df.sort_values(["household_id", period_key]).to_csv(scores_path, index=False)
    thresholds_df.sort_values(["household_id"]).to_csv(thresholds_path, index=False)
    alerts_df.sort_values(["household_id", period_key]).to_csv(alerts_path, index=False)
    sensor_profile_df.to_csv(sensor_profile_path, index=False)

    single_sensor_households = int(sensor_profile_df["single_sensor_only"].sum()) if not sensor_profile_df.empty else 0
    single_sensor_fraction = (
        float(sensor_profile_df["single_sensor_only"].mean()) if not sensor_profile_df.empty else 0.0
    )
    graph_signal_degraded = single_sensor_fraction >= 0.5

    summary = pd.DataFrame(
        {
            "n_households": [len(graphs_by_household)],
            "n_days_scored": [len(scores_df)],
            "n_alerts": [int(alerts_df["alert"].sum()) if not alerts_df.empty else 0],
            "alert_rate": [float(alerts_df["alert"].mean()) if not alerts_df.empty else 0.0],
            "target_alert_rate": [config.alert.target_alert_rate],
            "temporal_window": [config.data.temporal_window],
            "single_sensor_households": [single_sensor_households],
            "single_sensor_fraction": [single_sensor_fraction],
            "graph_signal_degraded": [graph_signal_degraded],
        }
    )
    summary_path = output_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "checkpoint": str(ckpt_path),
        "scores": str(scores_path),
        "thresholds": str(thresholds_path),
        "alerts": str(alerts_path),
        "sensor_profile": str(sensor_profile_path),
        "summary": str(summary_path),
    }
