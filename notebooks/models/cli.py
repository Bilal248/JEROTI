#!/usr/bin/env python3
"""
JEROTI CLI - Real-time process-based detection with online learning

Menu:
1. Manual options (train / collect)
2. Detect using trained model (on a CSV file)
3. Real-time detection & online learning from live OS processes
h. Help
q. Quit
"""

import os
import time
import argparse
from typing import Optional, Tuple

import psutil
import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ================== CONFIG ==================

BUILDS_DIR = "./builds"

# Teacher model (already trained, batch):
ISO_MODEL_PATH = os.path.join(BUILDS_DIR, "isolation_forest_model.pkl")
ISO_IMPUTER_PATH = os.path.join(BUILDS_DIR, "imputer.pkl")
ISO_SCALER_PATH = os.path.join(BUILDS_DIR, "scaler.pkl")

# Online learner to save:
ONLINE_MODEL_PATH = os.path.join(BUILDS_DIR, "jeroti_online_sgd.pkl")

# ================== UI HELPERS ==================

def print_header() -> None:
    print("==============================")
    print("        JEROTI CLI           ")
    print("==============================")
    print("1. Manual options (train / collect)")
    print("2. Detect using trained model")
    print("3. Start / Stop Real-time detection with online learning")
    print("h. Help")
    print("q. Quit")
    print("==============================")


def print_help() -> None:
    print("\nJEROTI CLI - Help")
    print("=================")
    print("1. Manual options (train / collect)")
    print("   - Connect to your training / data collection scripts.")
    print()
    print("2. Detect using trained model")
    print("   - Run one-shot detection on a CSV file using a saved model.")
    print()
    print("3. Real-time detection with online learning (process-based)")
    print("   - Periodically reads current OS processes via psutil.")
    print("   - Uses IsolationForest (teacher) to label processes.")
    print("   - Trains an online SGDClassifier with partial_fit on those labels.")
    print("   - Prints batch and global metrics vs teacher.")
    print()


# ================== DATA / FEATURES ==================

def get_process_snapshot() -> pd.DataFrame:
    """
    Collect a snapshot of current OS processes using psutil.
    Returns a DataFrame with columns similar to your dataset:
    pid, name, cpu, ram, timestamp, etc.
    """
    rows = []
    ts = pd.Timestamp.utcnow()

    # psutil.cpu_percent is stateful; call once to init, then again to get values.
    psutil.cpu_percent(interval=None)

    for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info"]):
        info = proc.info
        pid = info.get("pid", -1)
        name = info.get("name", "")
        cpu = info.get("cpu_percent", 0.0)
        mem_info = info.get("memory_info", None)
        if mem_info is not None:
            ram_mb = mem_info.rss / (1024 * 1024)
        else:
            ram_mb = 0.0

        row = {
            "timestamp": ts.isoformat(),
            "pid": float(pid),
            "name": name,
            "cpu": float(cpu),
            "ram": float(ram_mb),
            # Add placeholders or simple defaults for dataset-like columns:
            "occurrence_count": 1.0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply similar feature engineering as your training pipeline.
    """
    if "timestamp_parsed" not in df.columns and "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "timestamp_parsed" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp_parsed"], errors="coerce")

    if "last_seen" not in df.columns:
        # For real-time, last_seen = timestamp_parsed (single snapshot)
        df["last_seen"] = df.get("timestamp_parsed")

    if "hour" not in df.columns:
        df["hour"] = df["timestamp_parsed"].dt.hour
    if "day" not in df.columns:
        df["day"] = df["timestamp_parsed"].dt.day
    if "weekday" not in df.columns:
        df["weekday"] = df["timestamp_parsed"].dt.weekday
    if "time_alive" not in df.columns:
        # In real-time snapshot, you may not know lifetime; set 0 or NaN
        df["time_alive"] = (df["last_seen"] - df["timestamp_parsed"]).dt.total_seconds()

    return df


def select_numeric(df: pd.DataFrame) -> Tuple[np.ndarray, list]:
    num_df = df.select_dtypes(include=[np.number]).copy()
    return num_df.values, list(num_df.columns)


# ================== MANUAL OPTIONS (STUB) ==================

def manual_options_menu() -> None:
    while True:
        print("\nManual options:")
        print("1. Train models (placeholder)")
        print("2. Collect data (placeholder)")
        print("b. Back to main menu")
        c = input("Select an option: ").strip().lower()
        if c == "1":
            print("[Manual] Train models: plug your training scripts here.")
        elif c == "2":
            print("[Manual] Collect data: plug your collector here.")
        elif c == "b":
            break
        else:
            print("Invalid choice.")


# ================== LOAD TEACHER MODEL ==================

def load_isolation_forest_teacher():
    if not os.path.exists(ISO_MODEL_PATH):
        raise FileNotFoundError(f"IsolationForest model not found: {ISO_MODEL_PATH}")
    if not os.path.exists(ISO_IMPUTER_PATH) or not os.path.exists(ISO_SCALER_PATH):
        raise FileNotFoundError("Imputer or scaler for IsolationForest not found in ./builds")

    iso = joblib.load(ISO_MODEL_PATH)
    imputer: SimpleImputer = joblib.load(ISO_IMPUTER_PATH)
    scaler: StandardScaler = joblib.load(ISO_SCALER_PATH)
    return iso, imputer, scaler


# ================== OPTION 2: DETECT ON CSV ==================

def detect_using_trained_model(data_path: Optional[str] = None) -> None:
    if data_path is None:
        data_path = input("CSV path (default ../../data/all_data.csv): ").strip()
        if not data_path:
            data_path = "../../data/all_data.csv"

    if not os.path.exists(data_path):
        print(f"❌ File not found: {data_path}")
        return

    print(f"\n[Detect] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    X_raw, feature_cols = select_numeric(df)
    print(f"Using {len(feature_cols)} numeric features.")

    try:
        iso, imputer, scaler = load_isolation_forest_teacher()
    except Exception as e:
        print(f"❌ {e}")
        return

    X_cleaned = imputer.transform(X_raw)
    X_scaled = scaler.transform(X_cleaned)

    print("[Detect] Running IsolationForest inference...")
    preds = iso.predict(X_scaled)  # -1 anomaly, 1 normal
    df["is_anomaly"] = (preds == -1).astype(int)
    n_anom = (df["is_anomaly"] == 1).sum()
    print(f"Anomalies: {n_anom} / {len(df)} ({n_anom / len(df) * 100:.2f}%)")

    os.makedirs(BUILDS_DIR, exist_ok=True)
    out_path = os.path.join(BUILDS_DIR, "jeroti_detection_isolation_forest.csv")
    df.to_csv(out_path, index=False)
    print(f"[Detect] Results saved to: {out_path}")


# ================== OPTION 3: REAL-TIME + ONLINE LEARNING ==================

def realtime_process_detection_with_online_learning(poll_interval: float = 5.0) -> None:
    """
    Real-time:
    - Every `poll_interval` seconds:
      * reads current OS processes with psutil
      * engineers features
      * applies teacher IsolationForest to get pseudo-labels
      * trains an online SGDClassifier (partial_fit) on those labels
      * prints batch & global metrics vs teacher
    """
    try:
        iso, imputer, scaler = load_isolation_forest_teacher()
    except Exception as e:
        print(f"❌ {e}")
        return

    # Online classifier
    online_clf = SGDClassifier(loss="log_loss", random_state=42)
    online_initialized = False

    all_teacher_labels = []
    all_online_preds = []

    print("\n[Real-time] Starting process-based detection with online learning.")
    print("[Real-time] Teacher: IsolationForest")
    print(f"[Real-time] Poll interval: {poll_interval} seconds")
    print("[Real-time] Press Ctrl+C to stop.\n")

    try:
        while True:
            # 1) Get current process snapshot
            df_proc = get_process_snapshot()
            if df_proc.empty:
                time.sleep(poll_interval)
                continue

            # 2) Engineer features
            df_proc = engineer_features(df_proc)

            # 3) Numeric features + preprocess like training
            X_raw, _ = select_numeric(df_proc)
            X_cleaned = imputer.transform(X_raw)
            X_scaled = scaler.transform(X_cleaned)

            # 4) Teacher predictions -> pseudo-labels
            teacher_preds = iso.predict(X_scaled)      # -1 / 1
            y_pseudo = np.where(teacher_preds == -1, 1, 0)  # 1 = anomaly, 0 = normal

            # 5) Online learning
            if not online_initialized:
                online_clf.partial_fit(X_scaled, y_pseudo, classes=np.array([0, 1]))
                online_initialized = True
            else:
                online_clf.partial_fit(X_scaled, y_pseudo)

            # 6) Evaluate online model vs teacher on this batch
            online_batch_preds = online_clf.predict(X_scaled)

            all_teacher_labels.extend(y_pseudo.tolist())
            all_online_preds.extend(online_batch_preds.tolist())

            # Batch metrics
            b_acc = accuracy_score(y_pseudo, online_batch_preds)
            b_prec = precision_score(y_pseudo, online_batch_preds, zero_division=0)
            b_rec = recall_score(y_pseudo, online_batch_preds, zero_division=0)
            b_f1 = f1_score(y_pseudo, online_batch_preds, zero_division=0)

            # Global metrics
            g_acc = accuracy_score(all_teacher_labels, all_online_preds)
            g_prec = precision_score(all_teacher_labels, all_online_preds, zero_division=0)
            g_rec = recall_score(all_teacher_labels, all_online_preds, zero_division=0)
            g_f1 = f1_score(all_teacher_labels, all_online_preds, zero_division=0)

            n_anom_batch = (y_pseudo == 1).sum()
            print(f"[Real-time] Snapshot: {len(df_proc)} processes, "
                  f"{n_anom_batch} teacher anomalies "
                  f"({n_anom_batch / len(df_proc) * 100:.2f}%)")
            print(f"  Batch vs teacher -> Acc {b_acc:.4f}, Prec {b_prec:.4f}, "
                  f"Rec {b_rec:.4f}, F1 {b_f1:.4f}")
            print(f"  Global vs teacher -> Acc {g_acc:.4f}, Prec {g_prec:.4f}, "
                  f"Rec {g_rec:.4f}, F1 {g_f1:.4f}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[Real-time] Stopped by user.")
        os.makedirs(BUILDS_DIR, exist_ok=True)
        joblib.dump(online_clf, ONLINE_MODEL_PATH)
        print(f"[Real-time] Online SGDClassifier saved to: {ONLINE_MODEL_PATH}")


# ================== MAIN CLI LOOP ==================

def main() -> None:
    parser = argparse.ArgumentParser(description="JEROTI CLI")
    parser.add_argument("--once", action="store_true", help="Run one menu interaction then exit")
    args = parser.parse_args()

    while True:
        print_header()
        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            manual_options_menu()
        elif choice == "2":
            detect_using_trained_model()
        elif choice == "3":
            interval = input("Poll interval seconds (default 5): ").strip()
            try:
                poll = float(interval) if interval else 5.0
            except ValueError:
                poll = 5.0
            realtime_process_detection_with_online_learning(poll_interval=poll)
        elif choice == "h":
            print_help()
        elif choice == "q":
            print("Goodbye.")
            break
        else:
            print("Invalid choice, try again.")

        if args.once:
            break


if __name__ == "__main__":
    main()