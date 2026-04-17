"""Test client for the real inference server."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import requests

# Basic configuration
BASE_URL = "http://localhost:5000"
TEST_CHAT_ID = 12345


def random_mel_spectrogram(n_mels: int = 80, n_frames: int = 100) -> list[list[float]]:
    """Generate a random mel spectrogram for testing."""
    spec = np.random.randn(n_mels, n_frames).astype(np.float32)
    # Make it slightly more realistic: lower values, some structure
    spec = spec * 0.1 + 0.5
    return spec.tolist()


def test_health() -> None:
    """Test health endpoint."""
    print("[TEST] Health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Response: {response.json()}")
    assert response.status_code == 200


def test_status(chat_id: int) -> None:
    """Check user status."""
    print(f"[TEST] Status for user {chat_id}...")
    response = requests.get(f"{BASE_URL}/status/{chat_id}")
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Error: {response.status_code}")


def test_baseline_collection(chat_id: int, num_samples: int = 5) -> None:
    """Test baseline sample collection."""
    print(f"[TEST] Collecting {num_samples} baseline samples...")

    for i in range(num_samples):
        print(f"  Sample {i+1}/{num_samples}...", end=" ", flush=True)
        mel_spec = random_mel_spectrogram()
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"chat_id": chat_id, "mel_spectrogram": mel_spec},
        )

        if response.status_code != 200:
            print(f"ERROR: {response.status_code}")
            print(response.json())
            return

        result = response.json()
        print(f"voice_status={result['voice_status']}, samples={result['num_samples_collected']}")

        if result["is_trained"]:
            print(f"    Model trained! Threshold={result.get('message')}")
            break

        time.sleep(0.1)


def test_anomaly_detection(chat_id: int, num_tests: int = 3) -> None:
    """Test anomaly detection after training."""
    print(f"[TEST] Running {num_tests} anomaly detection tests...")

    for i in range(num_tests):
        print(f"  Test {i+1}/{num_tests}...", end=" ", flush=True)
        mel_spec = random_mel_spectrogram()
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"chat_id": chat_id, "mel_spectrogram": mel_spec},
        )

        if response.status_code != 200:
            print(f"ERROR: {response.status_code}")
            print(response.json())
            return

        result = response.json()
        status = result["voice_status"]
        score = result["anomaly_score"]
        print(f"status={status}, score={score:.4f}")

        time.sleep(0.1)


def test_retrain(chat_id: int) -> None:
    """Test retrain endpoint."""
    print(f"[TEST] Retraining model for user {chat_id}...")
    response = requests.post(
        f"{BASE_URL}/retrain",
        json={"chat_id": chat_id},
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("Real Inference Server Test Client")
    print("=" * 60)

    try:
        # Check server is running
        test_health()
        print()

        # Test baseline collection
        test_baseline_collection(TEST_CHAT_ID, num_samples=5)
        print()

        # Check status
        test_status(TEST_CHAT_ID)
        print()

        # Test anomaly detection
        test_anomaly_detection(TEST_CHAT_ID, num_tests=3)
        print()

        # Test retrain
        test_retrain(TEST_CHAT_ID)
        print()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to server. Is it running?")
        print(f"Make sure the server is running on {BASE_URL}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
