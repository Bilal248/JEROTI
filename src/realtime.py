import os
import time
import threading
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import psutil

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# CONFIG / PATHS
# -------------------------------------------------------------------

BUILDS_DIR = "./builds"

ISO_MODEL_PATH = os.path.join(BUILDS_DIR, "isolation_forest_model.pkl")
ISO_IMPUTER_PATH = os.path.join(BUILDS_DIR, "imputer.pkl")
ISO_SCALER_PATH = os.path.join(BUILDS_DIR, "scaler.pkl")

ONLINE_MODEL_PATH = os.path.join(BUILDS_DIR, "jeroti_online_sgd.pkl")

# Internal state for background loop
_stop_event: Optional[threading.Event] = None
_thread: Optional[threading.Thread] = None


# -------------------------------------------------------------------
# HELPER: Load IsolationForest + preprocessing (teacher)
# -------------------------------------------------------------------

def load_isolation_forest_teacher() -> Tuple[object, SimpleImputer, StandardScaler]:
    """
    Load the pre-trained IsolationForest and its preprocessing objects
    from ./builds. These must have been created by your training script.
    """
    if not os.path.exists(ISO_MODEL_PATH):
        raise FileNotFoundError(f"IsolationForest model not found: {ISO_MODEL_PATH}")
    if not os.path.exists(ISO_IMPUTER_PATH):
        raise FileNotFoundError(f"Imputer not found: {ISO_IMPUTER_PATH}")
    if not os.path.exists(ISO_SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {ISO_SCALER_PATH}")

    iso = joblib.load(ISO_MODEL_PATH)
    imputer: SimpleImputer = joblib.load(ISO_IMPUTER_PATH)
    scaler: StandardScaler = joblib.load(ISO_SCALER_PATH)

    return iso, imputer, scaler


# -------------------------------------------------------------------
# HELPER: Get live process snapshot from OS
# -------------------------------------------------------------------

def get_process_snapshot() -> pd.DataFrame:
    """
    Collect a snapshot of current OS processes using psutil.
    Returns a DataFrame similar in spirit to your dataset
    (pid, name, cpu, ram, timestamp, etc.).
    """
    rows = []
    ts = pd.Timestamp.utcnow()

    # Warm up cpu_percent
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

        rows.append(
            {
                "timestamp": ts.isoformat(),
                "pid": float(pid),
                "name": name,
                "cpu": float(cpu),
                "ram": float(ram_mb),
                # Placeholders for compatibility with your data schema
                "occurrence_count": 1.0,
            }
        )

    df = pd.DataFrame(rows)
    return df


# -------------------------------------------------------------------
# HELPER: Feature engineering (align with your training code)
# -------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply similar feature engineering as in your training pipeline:
    - parse timestamp
    - create hour, day, weekday, time_alive, last_seen
    """
    # timestamp_parsed
    if "timestamp_parsed" not in df.columns and "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "timestamp_parsed" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp_parsed"], errors="coerce")
    else:
        # If missing, create from now
        df["timestamp_parsed"] = pd.Timestamp.utcnow()

    # last_seen: for live snapshot, treat as current timestamp
    if "last_seen" not in df.columns:
        df["last_seen"] = df["timestamp_parsed"]

    if "hour" not in df.columns:
        df["hour"] = df["timestamp_parsed"].dt.hour
    if "day" not in df.columns:
        df["day"] = df["timestamp_parsed"].dt.day
    if "weekday" not in df.columns:
        df["weekday"] = df["timestamp_parsed"].dt.weekday
    if "time_alive" not in df.columns:
        df["time_alive"] = (df["last_seen"] - df["timestamp_parsed"]).dt.total_seconds()

    return df


# -------------------------------------------------------------------
# HELPER: Select numeric features
# -------------------------------------------------------------------

def select_numeric(df: pd.DataFrame):
    """
    Select numeric columns from df and return (X, feature_names).
    """
    num_df = df.select_dtypes(include=[np.number]).copy()
    return num_df.values, list(num_df.columns)


# -------------------------------------------------------------------
# CORE REAL-TIME LOOP (blocking)
# -------------------------------------------------------------------

def _realtime_loop(poll_interval: float, stop_event: threading.Event) -> None:
    """
    Internal blocking loop that does real-time detection and online learning
    until stop_event is set.
    """
    try:
        iso, imputer, scaler = load_isolation_forest_teacher()
    except Exception as e:
        print(f"❌ {e}")
        return

    # Online classifier (supports partial_fit)
    online_clf = SGDClassifier(loss="log_loss", random_state=42)
    online_initialized = False

    all_teacher_labels = []
    all_online_preds = []

    print("\n[Real-time] Starting process-based detection with online learning.")
    print("[Real-time] Teacher: IsolationForest")
    print(f"[Real-time] Poll interval: {poll_interval} seconds")
    print("[Real-time] Use stop_all() or Ctrl+C (if foreground) to stop.\n")

    try:
        while not stop_event.is_set():
            # 1) Get current process snapshot
            df_proc = get_process_snapshot()
            if df_proc.empty:
                # No processes? Wait and continue
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

            # Sleep with early exit if stop requested
            for _ in range(int(poll_interval * 10)):
                if stop_event.is_set():
                    break
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Real-time] Interrupted by user.")

    finally:
        print("\n[Real-time] Stopping and saving online model...")
        os.makedirs(BUILDS_DIR, exist_ok=True)
        joblib.dump(online_clf, ONLINE_MODEL_PATH)
        print(f"[Real-time] Online SGDClassifier saved to: {ONLINE_MODEL_PATH}")


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def realtime_process_detection_with_online_learning(poll_interval: float = 5.0) -> None:
    """
    Foreground (blocking) real-time detection + online learning.
    Useful if you just run this module directly.
    """
    stop_event = threading.Event()
    _realtime_loop(poll_interval=poll_interval, stop_event=stop_event)


def start_realtime_learning_and_detection(poll_interval: float = 5.0) -> None:
    """
    Start the real-time learning + detection loop in a background thread.
    If already running, does nothing.
    """
    global _stop_event, _thread

    if is_running():
        print("[Real-time] Already running.")
        return

    _stop_event = threading.Event()
    _thread = threading.Thread(
        target=_realtime_loop,
        args=(poll_interval, _stop_event),
        daemon=True,
    )
    _thread.start()
    print("[Real-time] Background real-time detection started.")


def stop_all() -> None:
    """
    Stop any running real-time learning + detection loop.
    """
    global _stop_event, _thread

    if _stop_event is None:
        print("[Real-time] No active run to stop.")
        return

    print("[Real-time] Stopping background loop...")
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=10.0)
    _stop_event = None
    _thread = None
    print("[Real-time] Stopped.")


def is_running() -> bool:
    """
    Return True if the real-time learning + detection loop is currently running.
    """
    global _thread
    return _thread is not None and _thread.is_alive()


if __name__ == "__main__":
    # If run directly, run in foreground (blocking)
    realtime_process_detection_with_online_learning(poll_interval=5.0)