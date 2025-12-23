import joblib
import random
import string
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import json
import numpy as np

def train_with_isolation_forest(X, feature_stats, dataset_names, MODEL_DIR):
    def generate_random_model_name(prefix="ISO_"):
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}{suffix}_{timestamp}"

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42
        ))
    ])
    pipe.fit(X)
    preds = pipe.predict(X)
    anomalies = int((preds == -1).sum())
    total = len(preds)
    performance = {
        "total_samples": total,
        "anomalies_detected": anomalies,
        "anomaly_fraction": round(anomalies / total, 4)
    }
    model_name = generate_random_model_name()
    model_file = os.path.join(MODEL_DIR, model_name + ".pkl")
    json_file = os.path.join(MODEL_DIR, model_name + ".json")
    joblib.dump(pipe, model_file)
    metadata = {
        "model_name": model_name,
        "model_type": "IsolationForest_PIPELINE",
        "model_file": model_file,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance": performance,
        "feature_stats": feature_stats,
        "datasets": dataset_names
    }
    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata