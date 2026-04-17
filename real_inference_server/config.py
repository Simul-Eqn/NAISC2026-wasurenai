"""Configuration for the real inference server."""

from pathlib import Path

# Basic Flask config
DEBUG = False
HOST = "127.0.0.1"
PORT = 5000

# Directories
REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = Path(__file__).resolve().parent
DATA_DIR = SERVER_DIR / "data"
MODELS_DIR = SERVER_DIR / "models"
METADATA_DIR = SERVER_DIR / "metadata"

# Baseline configuration
NUM_BASELINE_SAMPLES = 5  # Collect this many samples before training
TRAINING_EPOCHS = 10
TRAINING_LEARNING_RATE = 1e-3

# Anomaly detection thresholds
THRESHOLD_QUANTILE = 0.95  # Use this quantile of baseline scores as threshold
ANOMALY_THRESHOLD_MILD = None  # Will be computed as THRESHOLD_QUANTILE
ANOMALY_THRESHOLD_HIGH = None  # Will be set higher (e.g., 0.99 quantile or mean + 2*std)

# GPU Configuration
USE_GPU = True
GPU_DEVICE = "cuda"

# SVD Preprocessor
SVD_PREPROCESSOR_PATH = Path(REPO_ROOT) / "new_inference_server" / "voxpopuli" / "svd_preprocessor.npz"

# MERaLiON Model
MERALION_REPO_ID = "MERaLiON/MERaLiON-SpeechEncoder-2"

# Response levels
ANOMALY_LEVELS = {
    "normal": 0,
    "mild_anomaly": 1,
    "high_anomaly": 2,
}
