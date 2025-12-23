import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime
import random
import string
import glob

# We'll use an SGDClassifier with log loss (approximate online boosted classifier)
# It supports partial_fit for incremental updates.

CLASSIFIER_NAME = "ADA_ONLINE"

def generate_model_name(prefix="ADA_ONLINE_"):
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{prefix}{suffix}_{timestamp}"


def train_adaboost_with_iso_dbscan(X, iso_preds, db_preds, MODEL_DIR):
    # X is expected to be scaled features (numpy array)
    db_anomaly = np.array([1 if x == -1 else 0 for x in db_preds])
    iso_anomaly = np.array([1 if x == -1 else 0 for x in iso_preds])
    final_labels = np.where((db_anomaly == 1) | (iso_anomaly == 1), 1, 0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(MODEL_DIR, f"{CLASSIFIER_NAME}_*.pkl"))
    model = None
    scaler = None

    if existing:
        latest = max(existing, key=os.path.getmtime)
        try:
            saved = joblib.load(latest)
            model = saved.get("model")
            scaler = saved.get("scaler")
        except Exception:
            model = None
            scaler = None

    # If no existing model, create a fresh SGDClassifier
    if model is None:
        model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

    # Ensure X is numpy array and labels are ints
    X = np.asarray(X)
    y = np.asarray(final_labels).astype(int)

    # For first-time partial_fit we need to provide classes
    try:
        if not hasattr(model, "classes_"):
            model.partial_fit(X, y, classes=np.array([0, 1]))
        else:
            model.partial_fit(X, y)
    except Exception:
        # fallback to fit
        model.fit(X, y)

    # Save model and a trivial scaler placeholder (None) as dict
    model_name = generate_model_name()
    model_file = os.path.join(MODEL_DIR, model_name + ".pkl")
    meta_file = os.path.join(MODEL_DIR, model_name + ".json")

    joblib.dump({"model": model, "scaler": scaler}, model_file)

    metadata = {
        "model_name": model_name,
        "model_type": "AdaBoost_online_proxy",
        "model_file": model_file,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": int(len(X)),
        "num_anomalies": int(y.sum())
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


# incremental update API for live cycles
def partial_train_on_batch(X_batch, y_batch, MODEL_DIR):
    """X_batch: numpy array features (scaled), y_batch: binary labels"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(MODEL_DIR, f"{CLASSIFIER_NAME}_*.pkl"))
    model = None
    if existing:
        latest = max(existing, key=os.path.getmtime)
        try:
            saved = joblib.load(latest)
            model = saved.get("model")
        except Exception:
            model = None

    if model is None:
        model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)

    Xb = np.asarray(X_batch)
    yb = np.asarray(y_batch).astype(int)

    if not hasattr(model, "classes_"):
        model.partial_fit(Xb, yb, classes=np.array([0, 1]))
    else:
        model.partial_fit(Xb, yb)

    # Save updated model
    model_name = generate_model_name()
    model_file = os.path.join(MODEL_DIR, model_name + ".pkl")
    joblib.dump({"model": model, "scaler": None}, model_file)
    return model_file