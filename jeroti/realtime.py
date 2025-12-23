import os
import time
import threading
from typing import Tuple, Optional
import signal
from datetime import datetime

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

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Legacy artifacts (if you have them)
BUILDS_DIR = os.path.join(_BASE_DIR, "builds")
ISO_MODEL_PATH = os.path.join(BUILDS_DIR, "isolation_forest_model.pkl")
ISO_IMPUTER_PATH = os.path.join(BUILDS_DIR, "imputer.pkl")
ISO_SCALER_PATH = os.path.join(BUILDS_DIR, "scaler.pkl")

# Current training pipeline outputs go into ./model via train.py
MODEL_DIR = os.path.join(_BASE_DIR, "model")

ONLINE_MODEL_PATH = os.path.join(BUILDS_DIR, "jeroti_online_sgd.pkl")

# Internal state for background loop
_stop_event: Optional[threading.Event] = None
_thread: Optional[threading.Thread] = None

_sigint_installed = False


def _install_sigint_handler_once() -> None:
    """Ensure Ctrl+C stops the background loop cleanly."""
    global _sigint_installed
    if _sigint_installed:
        return

    def _handle_sigint(_sig, _frame):
        # Keep handler minimal; stop_all does safe joins/timeouts.
        try:
            stop_all()
        finally:
            raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
        _sigint_installed = True
    except Exception:
        # Some environments may disallow signal handlers; ignore.
        _sigint_installed = False


# -------------------------------------------------------------------
# HELPER: Load IsolationForest + preprocessing (teacher)
# -------------------------------------------------------------------

def _find_latest_isolation_pipeline(model_dir: str) -> Optional[str]:
    if not os.path.isdir(model_dir):
        return None

    candidates = []
    for name in os.listdir(model_dir):
        if not name.lower().endswith(".pkl"):
            continue
        # Training saves pipelines with ISO_* prefix for IsolationForest
        if name.startswith("ISO_"):
            candidates.append(os.path.join(model_dir, name))

    if not candidates:
        return None

    return max(candidates, key=lambda p: os.path.getmtime(p))


def load_isolation_forest_teacher() -> Tuple[object, Optional[SimpleImputer], Optional[StandardScaler]]:
    """Load a teacher model for pseudo-labeling.

    Supports two setups:
    1) Legacy "builds" artifacts: separate model + imputer + scaler.
    2) Current training output in ./model: a scikit-learn Pipeline saved by train.py.
    """
    # 1) Legacy builds/ format
    if os.path.exists(ISO_MODEL_PATH) and os.path.exists(ISO_IMPUTER_PATH) and os.path.exists(ISO_SCALER_PATH):
        iso = joblib.load(ISO_MODEL_PATH)
        imputer: SimpleImputer = joblib.load(ISO_IMPUTER_PATH)
        scaler: StandardScaler = joblib.load(ISO_SCALER_PATH)
        return iso, imputer, scaler

    # 2) Current ./model pipeline format
    latest_pipe = _find_latest_isolation_pipeline(MODEL_DIR)
    if latest_pipe is None:
        raise FileNotFoundError(
            "No IsolationForest teacher model found. "
            "Train one first (CLI -> Train new model -> Isolation Forest), "
            f"or place artifacts in {MODEL_DIR} (expected ISO_*.pkl)."
        )

    iso = joblib.load(latest_pipe)
    return iso, None, None


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

    for proc in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_info", "num_threads"]):
        try:
            info = proc.info
            pid = info.get("pid", -1)
            name = info.get("name", "") or ""

            # psutil can return None for cpu_percent depending on timing/platform
            cpu_val = info.get("cpu_percent", 0.0)
            cpu = float(cpu_val) if cpu_val is not None else 0.0

            mem_info = info.get("memory_info", None)
            rss_val = getattr(
                mem_info, "rss", 0) if mem_info is not None else 0
            rss = float(rss_val) if rss_val is not None else 0.0

            threads_val = info.get("num_threads", 0)
            threads = float(threads_val) if threads_val is not None else 0.0

            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "pid": float(pid) if pid is not None else -1.0,
                    "name": name,
                    "cpu": cpu,
                    # Align with your training columns used in train.py
                    "rss": rss,
                    "threads": threads,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        except Exception:
            # Keep the loop resilient: skip any unexpected per-process parsing issues
            continue

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
        df["timestamp_parsed"] = pd.to_datetime(
            df["timestamp"], errors="coerce")
    elif "timestamp_parsed" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(
            df["timestamp_parsed"], errors="coerce")
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
        df["time_alive"] = (
            df["last_seen"] - df["timestamp_parsed"]).dt.total_seconds()

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

def _realtime_loop(poll_interval: float, stop_event: threading.Event, actionmode: int = 2) -> None:
    """
    Internal blocking loop that does real-time detection and online learning
    until stop_event is set.
    """
    try:
        iso, imputer, scaler = load_isolation_forest_teacher()
    except Exception as e:
        print(f"[Real-time] {e}")
        return

    if actionmode not in (1, 2):
        actionmode = 2

    logs_dir = os.path.join(_BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(
        logs_dir, f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

            # 3) Features aligned with train.py
            X_raw = df_proc[["cpu", "rss", "threads"]].fillna(
                0.0).to_numpy(dtype=float)

            # If legacy preprocessing exists, apply it; otherwise the loaded Pipeline handles scaling.
            if imputer is not None and scaler is not None:
                X_cleaned = imputer.transform(X_raw)
                X_scaled = scaler.transform(X_cleaned)
            else:
                X_scaled = X_raw

            # 4) Teacher predictions -> pseudo-labels
            # -1 / 1 (Pipeline or model)
            teacher_preds = iso.predict(X_scaled)
            # 1 = anomaly, 0 = normal
            y_pseudo = np.where(teacher_preds == -1, 1, 0)

            # 5) Online learning
            if not online_initialized:
                online_clf.partial_fit(
                    X_scaled, y_pseudo, classes=np.array([0, 1]))
                online_initialized = True
            else:
                online_clf.partial_fit(X_scaled, y_pseudo)

            # 6) Evaluate online model vs teacher on this batch
            online_batch_preds = online_clf.predict(X_scaled)

            all_teacher_labels.extend(y_pseudo.tolist())
            all_online_preds.extend(online_batch_preds.tolist())

            # Batch metrics
            b_acc = accuracy_score(y_pseudo, online_batch_preds)
            b_prec = precision_score(
                y_pseudo, online_batch_preds, zero_division=0)
            b_rec = recall_score(y_pseudo, online_batch_preds, zero_division=0)
            b_f1 = f1_score(y_pseudo, online_batch_preds, zero_division=0)

            # Global metrics
            g_acc = accuracy_score(all_teacher_labels, all_online_preds)
            g_prec = precision_score(
                all_teacher_labels, all_online_preds, zero_division=0)
            g_rec = recall_score(all_teacher_labels,
                                 all_online_preds, zero_division=0)
            g_f1 = f1_score(all_teacher_labels,
                            all_online_preds, zero_division=0)

            n_anom_batch = int((y_pseudo == 1).sum())
            print(f"[Real-time] Snapshot: {len(df_proc)} processes, "
                  f"{n_anom_batch} teacher anomalies "
                  f"({n_anom_batch / len(df_proc) * 100:.2f}%)")
            print(f"  Batch vs teacher -> Acc {b_acc:.4f}, Prec {b_prec:.4f}, "
                  f"Rec {b_rec:.4f}, F1 {b_f1:.4f}")
            print(f"  Global vs teacher -> Acc {g_acc:.4f}, Prec {g_prec:.4f}, "
                  f"Rec {g_rec:.4f}, F1 {g_f1:.4f}")

            # 7) Action on anomalies: log (2) or kill top anomalous process (1)
            if n_anom_batch > 0:
                anomalous = df_proc.loc[y_pseudo == 1, [
                    "pid", "name", "cpu", "rss", "threads"]].copy()
                anomalous = anomalous.sort_values(by="cpu", ascending=False)

                top = anomalous.head(5)
                top_desc = ", ".join(
                    [
                        f"{int(r.pid)}:{r.name} cpu={float(r.cpu):.1f}"
                        for r in top.itertuples(index=False)
                    ]
                )
                msg = f"[ANOMALY] {n_anom_batch} anomalous processes. Top: {top_desc}"
                try:
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")
                except Exception:
                    pass

                if actionmode == 1 and not top.empty:
                    pid_to_kill = int(top.iloc[0]["pid"])
                    name_to_kill = str(top.iloc[0]["name"])
                    try:
                        psutil.Process(pid_to_kill).kill()
                        kill_msg = f"[ACTION] Killed PID={pid_to_kill} Name={name_to_kill}"
                        print(kill_msg)
                        with open(log_file, "a") as f:
                            f.write(kill_msg + "\n")
                    except Exception as e:
                        fail_msg = f"[ACTION] Failed to kill PID={pid_to_kill} Name={name_to_kill} -> {e}"
                        print(fail_msg)
                        try:
                            with open(log_file, "a") as f:
                                f.write(fail_msg + "\n")
                        except Exception:
                            pass

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
        print(
            f"[Real-time] Online SGDClassifier saved to: {ONLINE_MODEL_PATH}")


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def realtime_process_detection_with_online_learning(poll_interval: float = 5.0, actionmode: int = 2) -> None:
    """
    Foreground (blocking) real-time detection + online learning.
    Useful if you just run this module directly.
    """
    stop_event = threading.Event()
    _realtime_loop(poll_interval=poll_interval,
                   stop_event=stop_event, actionmode=actionmode)


def start_realtime_learning_and_detection(poll_interval: float = 5.0, actionmode: int = 2) -> None:
    """
    Start the real-time learning + detection loop in a background thread.
    If already running, does nothing.
    """
    global _stop_event, _thread

    if is_running():
        print("[Real-time] Already running.")
        return

    _install_sigint_handler_once()

    _stop_event = threading.Event()
    _thread = threading.Thread(
        target=_realtime_loop,
        args=(poll_interval, _stop_event, actionmode),
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
