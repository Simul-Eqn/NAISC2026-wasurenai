"""Flask inference server for personalized voice anomaly detection."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import HOST, NUM_BASELINE_SAMPLES, PORT
from data_manager import DataManager
from model_manager import ModelManager
from pipeline import ProcessingPipeline

# Initialize Flask app
app = Flask(__name__)

# Initialize managers
data_manager = DataManager()
model_manager = ModelManager()
pipeline = ProcessingPipeline()


def _validate_request() -> tuple[int, np.ndarray, str | None]:
    """Validate and extract mel spectrogram and chat_id from request.

    Returns:
        Tuple of (chat_id, mel_spectrogram, error_message)
    """
    data = request.get_json()
    if data is None:
        return None, None, "Request must be JSON"

    # Extract chat_id
    chat_id = data.get("chat_id")
    if chat_id is None:
        return None, None, "Missing 'chat_id' in request"

    try:
        chat_id = int(chat_id)
    except (ValueError, TypeError):
        return None, None, "chat_id must be an integer"

    # Extract mel_spectrogram
    mel_spec_list = data.get("mel_spectrogram")
    if mel_spec_list is None:
        return None, None, "Missing 'mel_spectrogram' in request"

    try:
        mel_spec = np.array(mel_spec_list, dtype=np.float32)
    except (ValueError, TypeError):
        return None, None, "mel_spectrogram must be a valid array"

    if mel_spec.ndim not in (1, 2):
        return None, None, "mel_spectrogram must be 1D or 2D array"

    return chat_id, mel_spec, None


@app.route("/analyze", methods=["POST"])
def analyze() -> dict[str, Any]:
    """Analyze voice for anomaly detection.

    Request JSON:
    {
        "chat_id": 12345,
        "mel_spectrogram": [[...], [...], ...]
    }

    Response:
    {
        "status": "success",
        "chat_id": 12345,
        "voice_status": "normal" | "mild_anomaly" | "high_anomaly",
        "anomaly_score": 0.5,
        "num_samples_collected": 3,
        "is_trained": false,
        "message": "..."
    }
    """
    chat_id, mel_spec, error = _validate_request()
    if error:
        return jsonify({"status": "error", "message": error}), 400

    try:
        # Store mel spectrogram
        filename = data_manager.save_mel_spectrogram(chat_id, mel_spec)

        # Get user metadata
        num_samples = data_manager.get_num_samples(chat_id)
        metadata = data_manager.get_user_metadata(chat_id)

        # Check if we have enough baseline samples
        if num_samples <= NUM_BASELINE_SAMPLES:
            # Still collecting baseline
            metadata = data_manager.update_user_metadata(
                chat_id,
                {
                    "num_samples": num_samples,
                    "last_filename": filename,
                },
            )
            return jsonify(
                {
                    "status": "success",
                    "chat_id": chat_id,
                    "voice_status": "normal",
                    "anomaly_score": None,
                    "num_samples_collected": num_samples,
                    "is_trained": False,
                    "message": f"Baseline collection: {num_samples}/{NUM_BASELINE_SAMPLES} samples. Model will train after {NUM_BASELINE_SAMPLES} samples.",
                }
            ), 200

        # Check if model is trained
        if not metadata.get("is_trained"):
            # Load all baseline samples and train
            mel_specs = data_manager.load_user_mel_spectrograms(chat_id)
            graphs = [
                pipeline.mel_spec_to_graph(spec, graph_id=f"{chat_id}_{i}")
                for i, (_, spec) in enumerate(mel_specs)
            ]

            # Train model
            threshold, mean_score = model_manager.train_model(chat_id, graphs)

            # Update metadata
            metadata = data_manager.update_user_metadata(
                chat_id,
                {
                    "num_samples": num_samples,
                    "is_trained": True,
                    "threshold": float(threshold),
                    "mean_score": float(mean_score),
                    "last_filename": filename,
                },
            )

            return jsonify(
                {
                    "status": "success",
                    "chat_id": chat_id,
                    "voice_status": "normal",
                    "anomaly_score": None,
                    "num_samples_collected": num_samples,
                    "is_trained": True,
                    "message": f"Model trained on {num_samples} samples. Threshold: {threshold:.4f}",
                }
            ), 200

        # Model is trained, score the new sample
        graph = pipeline.mel_spec_to_graph(mel_spec, graph_id=f"{chat_id}_{num_samples}")
        score, voice_status = model_manager.score_graph(chat_id, graph)

        # Update metadata
        metadata = data_manager.update_user_metadata(
            chat_id,
            {
                "num_samples": num_samples,
                "last_score": float(score),
                "last_filename": filename,
            },
        )

        return jsonify(
            {
                "status": "success",
                "chat_id": chat_id,
                "voice_status": voice_status,
                "anomaly_score": float(score),
                "num_samples_collected": num_samples,
                "is_trained": True,
                "message": f"Score: {score:.4f}",
            }
        ), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Internal error: {str(e)}"}), 500


@app.route("/retrain", methods=["POST"])
def retrain() -> dict[str, Any]:
    """Retrain a user's model treating all collected samples as normal.

    Request JSON:
    {
        "chat_id": 12345
    }

    Response:
    {
        "status": "success",
        "chat_id": 12345,
        "num_samples": 10,
        "threshold": 0.5,
        "message": "..."
    }
    """
    data = request.get_json()
    if data is None:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    chat_id = data.get("chat_id")
    if chat_id is None:
        return jsonify({"status": "error", "message": "Missing 'chat_id'"}), 400

    try:
        chat_id = int(chat_id)
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "chat_id must be an integer"}), 400

    try:
        # Load all mel spectrograms for the user
        mel_specs = data_manager.load_user_mel_spectrograms(chat_id)
        if not mel_specs:
            return jsonify(
                {
                    "status": "error",
                    "message": f"No samples found for user {chat_id}",
                }
            ), 404

        # Convert to graphs
        graphs = [
            pipeline.mel_spec_to_graph(spec, graph_id=f"{chat_id}_{i}")
            for i, (_, spec) in enumerate(mel_specs)
        ]

        # Train model
        threshold, mean_score = model_manager.train_model(chat_id, graphs)

        # Update metadata
        num_samples = len(mel_specs)
        data_manager.update_user_metadata(
            chat_id,
            {
                "num_samples": num_samples,
                "is_trained": True,
                "threshold": float(threshold),
                "mean_score": float(mean_score),
            },
        )

        return jsonify(
            {
                "status": "success",
                "chat_id": chat_id,
                "num_samples": num_samples,
                "threshold": float(threshold),
                "mean_score": float(mean_score),
                "message": f"Model retrained on {num_samples} samples",
            }
        ), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Internal error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health() -> dict[str, Any]:
    """Health check endpoint."""
    return jsonify(
        {
            "status": "ok",
            "device": model_manager.device,
            "svd_available": pipeline.svd_preprocessor is not None,
        }
    ), 200


@app.route("/status/<int:chat_id>", methods=["GET"])
def status(chat_id: int) -> dict[str, Any]:
    """Get status for a specific user."""
    try:
        metadata = data_manager.get_user_metadata(chat_id)
        has_model = model_manager.has_model(chat_id)

        return jsonify(
            {
                "status": "success",
                "chat_id": chat_id,
                "num_samples": metadata.get("num_samples", 0),
                "is_trained": metadata.get("is_trained", False),
                "has_model": has_model,
                "threshold": metadata.get("threshold"),
                "mean_score": metadata.get("mean_score"),
            }
        ), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print(f"Starting Flask server on {HOST}:{PORT}")
    print(f"Device: {model_manager.device}")
    print(f"Baseline samples before training: {NUM_BASELINE_SAMPLES}")
    print(f"SVD preprocessor available: {pipeline.svd_preprocessor is not None}")
    app.run(host=HOST, port=PORT, debug=False)
