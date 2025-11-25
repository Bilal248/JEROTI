import joblib
import random
import string
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import json
import numpy as np


def train_with_dbscan(X_scaled, feature_stats, dataset_names, MODEL_DIR):

    def generate_random_model_name(prefix="DBSCAN_"):
        suffix = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=6))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}{suffix}_{timestamp}"

    # --------------------------
    # 🔥 BUILD A PIPELINE
    # --------------------------
    scaler = StandardScaler()  # instantiate inside function

    pipe = Pipeline([
        ("scaler", scaler),
        ("dbscan", DBSCAN(eps=0.6, min_samples=10))
    ])

    labels = pipe.fit_predict(X_scaled)

    # --------------------------
    # METRICS
    # --------------------------
    anomalies = int(np.sum(labels == -1))
    total = len(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    performance = {
        "total_samples": total,
        "anomalies_detected": anomalies,
        "anomaly_fraction": round(anomalies / total, 4),
        "num_clusters": num_clusters
    }

    # --------------------------
    # SAVE MODEL + METADATA
    # --------------------------
    model_name = generate_random_model_name()
    model_file = os.path.join(MODEL_DIR, model_name + ".pkl")
    json_file = os.path.join(MODEL_DIR, model_name + ".json")

    # Saves both SCALER + DBSCAN together
    joblib.dump(pipe, model_file)

    metadata = {
        "model_name": model_name,
        "model_type": "DBSCAN_PIPELINE",
        "model_file": model_file,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance": performance,
        "feature_stats": feature_stats,
        "datasets": dataset_names
    }

    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
