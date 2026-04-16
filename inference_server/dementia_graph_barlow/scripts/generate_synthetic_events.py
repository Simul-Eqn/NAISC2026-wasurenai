from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic household sensor events")
    parser.add_argument("--output_csv", required=True, help="Path to output CSV")
    parser.add_argument("--households", type=int, default=3, help="Number of households")
    parser.add_argument("--days", type=int, default=30, help="Number of days")
    parser.add_argument("--sensors_per_household", type=int, default=4, help="Distinct sensors per household")
    parser.add_argument("--events_per_day", type=int, default=120, help="Average events per household/day")
    parser.add_argument("--anomaly_days", type=int, default=3, help="Anomalous days per household")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    rng = np.random.default_rng(args.seed)
    start_day = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows: list[dict] = []

    for h in range(args.households):
        household_id = f"H{h+1:02d}"
        sensor_ids = [f"{household_id}_S{i+1:02d}" for i in range(args.sensors_per_household)]
        anomaly_idx = set(rng.choice(args.days, size=min(args.anomaly_days, args.days), replace=False).tolist())

        for d in range(args.days):
            day_start = start_day + timedelta(days=d)
            event_count = max(10, int(rng.normal(args.events_per_day, args.events_per_day * 0.15)))

            is_anomaly_day = d in anomaly_idx
            if is_anomaly_day:
                event_count = max(8, int(event_count * rng.uniform(0.4, 1.8)))

            if args.sensors_per_household == 1:
                probs = np.array([1.0], dtype=np.float64)
            else:
                probs = np.ones(args.sensors_per_household, dtype=np.float64)
                if is_anomaly_day:
                    spike_idx = int(rng.integers(0, args.sensors_per_household))
                    probs[spike_idx] *= float(rng.uniform(2.5, 4.0))
                probs = probs / probs.sum()

            chosen_sensors = rng.choice(sensor_ids, size=event_count, p=probs)
            minutes = np.sort(rng.integers(0, 24 * 60, size=event_count))

            if is_anomaly_day:
                shift = int(rng.integers(-180, 180))
                minutes = np.clip(minutes + shift, 0, 24 * 60 - 1)

            for sensor_id, minute in zip(chosen_sensors, minutes):
                timestamp = day_start + timedelta(minutes=int(minute))
                rows.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "household_id": household_id,
                        "sensor_id": sensor_id,
                        "is_synthetic_anomaly_day": bool(is_anomaly_day),
                    }
                )

    output = pd.DataFrame(rows).sort_values(["household_id", "timestamp"]).reset_index(drop=True)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(
        f"Generated {len(output)} events for {args.households} households over {args.days} days "
        f"with {args.sensors_per_household} sensor(s)/household at {output_path}"
    )


if __name__ == "__main__":
    main()
