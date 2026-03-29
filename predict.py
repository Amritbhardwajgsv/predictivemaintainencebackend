#!/usr/bin/env python3
"""
predict.py  —  reads JSON from stdin, runs pickle model, prints JSON to stdout.
Model is downloaded from Google Drive at startup (NOT during request).
"""

import sys, json, pickle, os, numpy as np
import gdown

FEATURE_COLS = [
    "setting1", "setting2",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17",
    "s18", "s19",
    "s20", "s21"
]

BASE = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(BASE, "best_model.pkl")
SCALER_PATH = os.path.join(BASE, "scaler.pkl")

# 🔥 Google Drive model ID
MODEL_ID = "1jNzIlMOyBe3OD8s2iJL6E0PeTZhlvUM6"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...", file=sys.stderr)
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ✅ IMPORTANT: Download model at startup (only once)
ensure_model()

def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def status(rul):
    if rul <= 15:  return "CRITICAL"
    if rul <= 40:  return "WARNING"
    if rul <= 80:  return "MONITOR"
    return "HEALTHY"

def main():
    data = json.loads(sys.stdin.read().strip())

    features = np.array([[float(data[c]) for c in FEATURE_COLS]])

    # Debug logs
    print(f"Input data keys: {list(data.keys())}", file=sys.stderr)
    print(f"Feature columns: {FEATURE_COLS}", file=sys.stderr)
    print(f"Features array shape: {features.shape}", file=sys.stderr)

    model = load(MODEL_PATH)
    if model is None:
        rul = max(0, int(150 - abs(float(data["s2"])) * 0.1))
        print(json.dumps({
            "rul_cycles": rul,
            "status": status(rul),
            "warning": "best_model.pkl not found — demo prediction"
        }))
        return

    print("Expected features:", model.n_features_in_, file=sys.stderr)

    scaler = load(SCALER_PATH)
    if scaler:
        features = scaler.transform(features)
        print("Scaled features:", features, file=sys.stderr)

    pred = model.predict(features)[0]
    print("Raw prediction:", pred, file=sys.stderr)

    # Scale prediction (assuming normalized output)
    SCALE_FACTOR = 200
    rul = max(0, round(float(pred) * SCALE_FACTOR))

    result = {
        "rul_cycles": rul,
        "status": status(rul),
        "debug": {
            "expected_features": int(model.n_features_in_),
            "features_sent": len(FEATURE_COLS),
            "raw_prediction": float(pred),
            "scaled_rul": float(pred) * SCALE_FACTOR
        }
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()