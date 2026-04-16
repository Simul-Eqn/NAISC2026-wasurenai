# Dementia Graph Barlow Twins (Inference Prototype)

This folder implements a lightweight, domain-agnostic prototype of:

## Paper

Graph Contrastive Learning for Anomaly Detection and Personalized Alerting in Sensor-based Remote Monitoring for Dementia Care

The implementation focuses on:

- building daily household movement graphs from raw sensor events,
- self-supervised Graph Barlow Twins with aggregation-based node feature masking,
- anomaly scoring via day-to-day graph embedding drift,
- household-personalized thresholds using clinician target alert rate,
- explainability via node attention weights (top sensor contribution per alert).

## Folder Structure

- `src/data.py`: raw event loading and daily graph construction
- `src/model.py`: graph attention encoder + Barlow Twins objective
- `src/train.py`: self-supervised training and graph encoding
- `src/anomaly.py`: drift score, personalized thresholding, alerting
- `src/pipeline.py`: end-to-end training + inference + export
- `src/cli.py`: command-line entrypoint
- `configs/default.yaml`: default pipeline configuration

## Expected Input CSV

The pipeline expects at least these columns:

- `timestamp` (ISO date-time)
- `household_id`
- `sensor_id`

Optional raw-value column:

- `sensor_value` (or any numeric column mapped via `data.sensor_value_col` in config)

Raw-value mode:

- `data.sensor_value_mode: scalar` (default): scalar values or vectors reduced to row-level RMS, then aggregated per sensor-period
- `data.sensor_value_mode: vector`: fixed-length vectors are kept dimension-wise (no per-row scalar reduction), then aggregated per sensor-period as vector mean/std

When `sensor_value_col` is set and present, per-sensor daily value mean/std are added to node features.
The value can be either:

- a scalar numeric value per row, or
- a variable-length vector serialized as JSON list string (for example, `[0.12, -0.03, ...]`).

For `vector` mode, vector length must be fixed across rows. Rows with mismatched vector length are ignored for value-feature aggregation.

Temporal granularity is configurable in `configs/default.yaml`:

- `data.temporal_window: day` (default)
- `data.temporal_window: week`

For weekly mode, each graph corresponds to one household-week (week start date is used as the period key).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.cli \
  --events_csv path/to/events.csv \
  --config configs/default.yaml \
  --output_dir outputs/run_001
```

## Generate Synthetic Data

```bash
python scripts/generate_synthetic_events.py \
  --output_csv outputs/synthetic_events.csv \
  --households 5 \
  --days 45 \
  --sensors_per_household 4 \
  --events_per_day 120 \
  --anomaly_days 5
```

Single-sensor stress case:

```bash
python scripts/generate_synthetic_events.py \
  --output_csv outputs/synthetic_events_1sensor.csv \
  --households 5 \
  --days 45 \
  --sensors_per_household 1
```

## Outputs

- `graph_barlow_twins.pt`: trained self-supervised model checkpoint
- `daily_scores.csv`: per-household daily anomaly scores
- `daily_scores.csv`: per-household anomaly scores per selected temporal window (`day` or `week`)
- `personalized_thresholds.csv`: per-household threshold from calibration window
- `alerts.csv`: final alert decisions
- `household_sensor_profile.csv`: number of sensors per household and one-sensor flag
- `summary.csv`: run-level metrics including achieved alert rate

## One-sensor Evaluation

If every household has only one sensor:

- the graph collapses to one node with a self-loop, so topology/transition information is unavailable,
- anomaly signal mainly comes from temporal intensity shifts (event count/time-of-day changes),
- attention explainability becomes trivial (`top_sensor` is always that single sensor),
- personalized alerting still works, but sensitivity to spatial behavior change is reduced.

The pipeline explicitly reports this condition in `summary.csv`:

- `single_sensor_households`
- `single_sensor_fraction`
- `graph_signal_degraded` (true when at least half of households are single-sensor)

## Notes

- This is an inference-oriented implementation scaffold intended for extension with your exact study protocol and dataset conventions.
- The implementation avoids negative sampling and uses contrastive alignment based on two masked views of each daily graph.
