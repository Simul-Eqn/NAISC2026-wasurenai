from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _distance(prev: np.ndarray, curr: np.ndarray, metric: str = "cosine") -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(prev) * np.linalg.norm(curr)) + 1e-9
        return float(1.0 - np.dot(prev, curr) / denom)
    if metric == "l2":
        return float(np.linalg.norm(curr - prev))
    raise ValueError(f"Unsupported metric: {metric}")


def compute_daily_drift_scores(encoded: Dict[str, List[dict]], metric: str = "cosine") -> pd.DataFrame:
    rows = []
    for household_id, records in encoded.items():
        if not records:
            continue
        period_key = "period_start" if "period_start" in records[0] else "day"
        records_sorted = sorted(records, key=lambda item: item[period_key])
        for i in range(1, len(records_sorted)):
            prev = records_sorted[i - 1]
            curr = records_sorted[i]
            score = _distance(prev["embedding"], curr["embedding"], metric=metric)
            rows.append(
                {
                    "household_id": household_id,
                    "period_start": curr[period_key],
                    "day": curr[period_key],
                    "score": score,
                    "top_sensor": curr["sensor_names"][int(np.argmax(curr["node_weights"]))],
                    "top_weight": float(np.max(curr["node_weights"])),
                }
            )
    return pd.DataFrame(rows)


def fit_personalized_thresholds(scores_df: pd.DataFrame, target_alert_rate: float, calibration_ratio: float) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame(columns=["household_id", "threshold", "n_calibration_days"])

    period_key = "period_start" if "period_start" in scores_df.columns else "day"
    outputs = []
    for household_id, hh in scores_df.groupby("household_id"):
        hh = hh.sort_values(period_key)
        n_cal = max(1, int(round(len(hh) * calibration_ratio)))
        calibration = hh.iloc[:n_cal]

        quantile = float(1.0 - target_alert_rate)
        threshold = float(calibration["score"].quantile(quantile))
        outputs.append(
            {
                "household_id": household_id,
                "threshold": threshold,
                "n_calibration_days": int(n_cal),
            }
        )

    return pd.DataFrame(outputs)


def apply_alerting(scores_df: pd.DataFrame, thresholds_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty:
        return scores_df.assign(threshold=np.nan, alert=False)

    merged = scores_df.merge(thresholds_df, on="household_id", how="left")
    merged["alert"] = merged["score"] > merged["threshold"]
    return merged
