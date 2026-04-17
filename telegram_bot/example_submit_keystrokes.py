import requests

BASE_URL = "http://localhost:5000"
API_URL = f"{BASE_URL}/api/keystroke-checkin"

payload = {
    "patient_keystroke_id": "simuyay",
    "status": "mild_anomaly",      # normal | mild_anomaly | high_anomaly
    "anomaly_score": 0.67
}

headers = {
    "Content-Type": "application/json",
    # Include this only if KEYSTROKE_API_KEY is set on the server:
    # "X-API-Key": "your_api_key_here",
}

resp = requests.post(API_URL, json=payload, headers=headers, timeout=10)

print("status_code:", resp.status_code)
try:
    print("json:", resp.json())
except ValueError:
    print("text:", resp.text)