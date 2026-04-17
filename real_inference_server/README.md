# Real Inference Server

A Flask-based personalized voice anomaly detection server that uses graph-based anomaly detection with self-supervised learning.

## Overview

This server provides an HTTP API for real-time voice anomaly detection. It:

1. **Collects baseline samples** - Accepts mel spectrograms from users until a threshold is reached
2. **Trains personalized models** - After baseline collection, trains a Graph Barlow Twins model on a per-user basis
3. **Detects anomalies** - Scores new voice samples into three categories:
   - **normal**: Score below user's threshold
   - **mild_anomaly**: Score between threshold and 1.5× threshold
   - **high_anomaly**: Score above 1.5× threshold
4. **Retrains on demand** - Provides a `/retrain` endpoint to update models with new data

## Architecture

```
mel_spectrogram (input)
         ↓
  [MERaLiON encoder] (transformers)
         ↓
   embeddings [T, embedding_dim]
         ↓
   [SVD preprocessor] (optional dimensionality reduction)
         ↓
   embeddings [T, svd_dim]
         ↓
 [Graph construction] (sequential adjacency + self-loops)
         ↓
   GraphSample (nodes=timesteps, edges=temporal)
         ↓
 [Graph Barlow Twins model]
         ↓
  anomaly_score (cosine distance from center)
         ↓
voice_status (normal/mild_anomaly/high_anomaly)
```

## Configuration

Edit `config.py` to customize:

- `NUM_BASELINE_SAMPLES`: Number of samples to collect before training (default: 5)
- `TRAINING_EPOCHS`: Epochs for model training (default: 10)
- `USE_GPU`: Enable CUDA if available (default: True)
- `SVD_PREPROCESSOR_PATH`: Path to pre-fitted SVD model
- `MERALION_REPO_ID`: Hugging Face model ID for speech encoding

## API Endpoints

### POST `/analyze`

Analyze a voice sample and return anomaly detection result.

**Request:**
```json
{
  "chat_id": 12345,
  "mel_spectrogram": [
    [0.1, 0.2, ...],
    [0.15, 0.25, ...],
    ...
  ]
}
```

**Response (baseline collection):**
```json
{
  "status": "success",
  "chat_id": 12345,
  "voice_status": "normal",
  "anomaly_score": null,
  "num_samples_collected": 3,
  "is_trained": false,
  "message": "Baseline collection: 3/5 samples..."
}
```

**Response (after training):**
```json
{
  "status": "success",
  "chat_id": 12345,
  "voice_status": "mild_anomaly",
  "anomaly_score": 0.75,
  "num_samples_collected": 25,
  "is_trained": true,
  "message": "Score: 0.7500"
}
```

### POST `/retrain`

Retrain a user's model using all collected samples (useful for feedback loops).

**Request:**
```json
{
  "chat_id": 12345
}
```

**Response:**
```json
{
  "status": "success",
  "chat_id": 12345,
  "num_samples": 25,
  "threshold": 0.65,
  "mean_score": 0.52,
  "message": "Model retrained on 25 samples"
}
```

### GET `/health`

Health check and server status.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "svd_available": true
}
```

### GET `/status/<chat_id>`

Get user status and model metadata.

**Response:**
```json
{
  "status": "success",
  "chat_id": 12345,
  "num_samples": 25,
  "is_trained": true,
  "has_model": true,
  "threshold": 0.65,
  "mean_score": 0.52
}
```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the SVD preprocessor exists at the configured path:
   ```bash
   python ../new_inference_server/voxpopuli/svd_preprocessor.py \
     --data_root ../new_inference_server/voxpopuli/data \
     --output_path ./svd_preprocessor.npz
   ```

3. (Optional) Verify CUDA availability for GPU support.

## Running the Server

### Development (with automatic reload):
```bash
flask run
```

### Production (Gunicorn recommended):
```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

**Note:** The server uses a single worker (`-w 1`) because model inference maintains GPU state.

## Data Storage

All data is stored locally:

- **Mel spectrograms:** `data/<chat_id>/mel_spec_*.npy`
- **User metadata:** `metadata/<chat_id>_metadata.json`
- **Trained models:** `models/model_<chat_id>.pt`
- **Thresholds:** `models/threshold_<chat_id>.npy`

## Example Usage

```python
import requests
import numpy as np

BASE_URL = "http://localhost:5000"

# Create a test mel spectrogram
mel_spec = np.random.randn(80, 100).tolist()

# First 5 samples: baseline collection
for i in range(5):
    response = requests.post(
        f"{BASE_URL}/analyze",
        json={
            "chat_id": 12345,
            "mel_spectrogram": mel_spec,
        }
    )
    print(f"Sample {i+1}: {response.json()}")

# After 5 samples, model is trained
response = requests.post(
    f"{BASE_URL}/analyze",
    json={
        "chat_id": 12345,
        "mel_spectrogram": mel_spec,
    }
)
print(f"Anomaly detection: {response.json()}")
```

## Performance

- **GPU Memory:** ~2-4 GB (MERaLiON model + 1 inference)
- **Inference Time:** ~100-200ms per sample (mel spec → score)
- **Training Time:** ~1-2 minutes for 50 samples (depends on sample length)

## Troubleshooting

### "SVD preprocessor not found"
Make sure `SVD_PREPROCESSOR_PATH` in `config.py` points to a valid `.npz` file from `svd_preprocessor.py`.

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in pipeline
- Set `USE_GPU = False` in `config.py` to use CPU
- Monitor GPU with `nvidia-smi`

### Model not training
Ensure at least `NUM_BASELINE_SAMPLES` have been submitted for a user before the model trains.

## Architecture Decisions

- **Self-supervised learning:** Graph Barlow Twins objective requires no labels
- **Per-user models:** Each user gets a personalized threshold and anomaly center
- **Sequential graphs:** Timesteps form nodes, temporal adjacency forms edges
- **GPU-first:** All tensor operations stay on GPU; minimal CPU copies

## Future Improvements

- Multi-worker scaling with shared model caching
- Online/incremental learning without full retraining
- Explainability: node importance for anomaly decisions
- Threshold adaptation based on user feedback
