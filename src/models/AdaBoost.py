import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import json
import os
from datetime import datetime
import random
import string

def train_adaboost_with_iso_dbscan(X, iso_preds, db_preds, MODEL_DIR):
    db_anomaly = np.array([1 if x == -1 else 0 for x in db_preds])
    iso_anomaly = np.array([1 if x == -1 else 0 for x in iso_preds])
    final_labels = np.where((db_anomaly == 1) | (iso_anomaly == 1), 1, 0)

    base = DecisionTreeClassifier(max_depth=3)
    ada = AdaBoostClassifier(
        estimator=base,
        n_estimators=50,
        learning_rate=0.5,
        random_state=42
    )
    ada.fit(X, final_labels)

    def generate_model_name(prefix="ADA_"):
        suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}{suffix}_{timestamp}"

    model_name = generate_model_name()
    model_file = os.path.join(MODEL_DIR, model_name + ".pkl")
    meta_file = os.path.join(MODEL_DIR, model_name + ".json")
    joblib.dump(ada, model_file)
    metadata = {
        "model_name": model_name,
        "model_type": "AdaBoost_from_ISO_DBSCAN",
        "model_file": model_file,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(X),
        "num_anomalies": int(final_labels.sum())
    }
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata