from datetime import datetime
import pandas as pd
import os
from models.DBSA import train_with_dbscan
from models.IsolationForest import train_with_isolation_forest


def train_model(df=None, dataset_names=[], model_choice="auto"):

    MODEL_DIR = "model"
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------------------------
    # Load dataset
    # ---------------------------
    if df is None:
        raw_dir = "data/"
        files = [os.path.join(raw_dir, f)
                 for f in os.listdir(raw_dir) if f.endswith(".csv")]
        if not files:
            print("[train] No training data found")
            return

        df_list = [pd.read_csv(f) for f in files]
        df = pd.concat(df_list, ignore_index=True)

    df = df.fillna(0.0)
    X = df[["cpu", "rss", "threads"]]

    # ---------------------------
    # Feature stats
    # ---------------------------
    feature_stats = {
        col: {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "z_scores_sample": ((X[col] - X[col].mean()) / X[col].std()).head(5).tolist()
        } for col in X.columns
    }

    # ---------------------------
    # MODEL CHOICE
    # ---------------------------
    if model_choice == "isolation":
        return {
            "selected_model": train_with_isolation_forest(
                X, feature_stats, dataset_names, MODEL_DIR
            )
        }

    if model_choice == "dbscan":
        return {
            "selected_model": train_with_dbscan(
                X, feature_stats, dataset_names, MODEL_DIR
            )
        }

    # ---------------------------
    # AUTO TRAIN BOTH MODELS
    # ---------------------------
    print("[AUTO] Training both IsolationForest and DBSCAN ...")

    dbscan_meta = train_with_dbscan(
        X, feature_stats, dataset_names, MODEL_DIR
    )
    iso_meta = train_with_isolation_forest(
        X, feature_stats, dataset_names, MODEL_DIR
    )

    # ---------------------------
    # SELECT BEST MODEL
    # ---------------------------
    def score(model_meta):
        return 1 - model_meta["performance"]["anomaly_fraction"]

    iso_score = score(iso_meta)
    dbscan_score = score(dbscan_meta)

    if iso_score >= dbscan_score:
        best = iso_meta
        print("[AUTO] Isolation Forest selected as best model.")
    else:
        best = dbscan_meta
        print("[AUTO] DBSCAN selected as best model.")

    return {
        "isolation_forest": iso_meta,
        "dbscan": dbscan_meta,
        "best_model": best

    }