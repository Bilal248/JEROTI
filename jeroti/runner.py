# runner.py
import time
import threading
import os
import psutil
import joblib
import numpy as np
import copy
import signal
import warnings
from datetime import datetime
from train import train_model

_running_detector = False
_stop_flag = False

# Rolling buffer for online learning
_online_buffer = []
ONLINE_BUFFER_LIMIT = 100  # After 100 samples → retrain
_online_lock = threading.Lock()  # Lock for buffer


# -------------------- Signal Handler --------------------
def signal_handler(sig, frame):
    global _stop_flag
    print("\n[signal] Ctrl+C detected → stopping all real-time processes...")
    _stop_flag = True


signal.signal(signal.SIGINT, signal_handler)


# -------------------- System Data Collection --------------------
def _collect_system_sample():
    cpu = psutil.cpu_percent(interval=None)
    rss = psutil.virtual_memory().used / (1024 * 1024)  # MB
    threads = threading.active_count()

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": cpu,
        "rss": rss,
        "threads": threads
    }


# -------------------- Detection Logic --------------------
def _detect_from_model(model, scaler, sample):
    X = np.array([[sample["cpu"], sample["rss"], sample["threads"]]])

    if scaler:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # Isolation Forest / Pipeline
    if hasattr(model, "predict") and not hasattr(model, "labels_"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = model.predict(X_scaled)[0]
        return pred == -1

    # DBSCAN (approximate)
    if hasattr(model, "labels_") and hasattr(model, "components_"):
        from sklearn.neighbors import NearestNeighbors
        try:
            nbrs = NearestNeighbors(n_neighbors=1).fit(model.components_)
            _, _ = nbrs.kneighbors(X_scaled)
            return False
        except:
            return True

    return False


# -------------------- Detector Loop --------------------
def _detector_loop(model_file_ref, actionmode=2):
    global _running_detector, _stop_flag

    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join(
        "logs", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    current_model_file = None
    model = None
    scaler = None

    print("[detector] Detector loop started.")

    while not _stop_flag:
        # Reload model if updated
        if model_file_ref['file'] != current_model_file:
            current_model_file = model_file_ref['file']
            if current_model_file is None:
                time.sleep(1)
                continue
            saved = joblib.load(current_model_file)
            if isinstance(saved, dict):
                model = copy.deepcopy(saved.get("model", None))
                scaler = copy.deepcopy(saved.get("scaler", None))
            else:
                model = copy.deepcopy(saved)
                scaler = None

            print(
                f"[detector] Loaded model: {os.path.basename(current_model_file)}")

        if model is None:
            time.sleep(1)
            continue

        sample = _collect_system_sample()
        is_anomaly = _detect_from_model(model, scaler, sample)

        if is_anomaly:
            top_procs = sorted(
                psutil.process_iter(["name", "cpu_percent"]),
                key=lambda p: p.info.get("cpu_percent") or 0,
                reverse=True
            )
            proc_names = [p.info["name"] for p in top_procs[:5]
                          if p.info.get("cpu_percent") is not None and p.info["cpu_percent"] > 0]

            log_entry = f"[!!] Anomaly Detected → Processes: {', '.join(proc_names)}"
            print(log_entry)
            with open(log_file, "a") as f:
                f.write(log_entry + "\n")

            if actionmode == 1 and top_procs:
                proc_to_kill = top_procs[0]
                try:
                    proc_to_kill.kill()
                    msg = f"[ACTION] Killed process PID={proc_to_kill.pid}, Name={proc_to_kill.info['name']}"
                    print(msg)
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    msg = f"[ACTION] Failed to kill PID={proc_to_kill.pid} → {e}"
                    print(msg)
                    with open(log_file, "a") as f:
                        f.write(msg + "\n")

        time.sleep(1)

    print("[detector] Stopped.")
    _running_detector = False


def run_detection_with_saved_model(model_file, actionmode=2):
    global _running_detector, _stop_flag

    if not os.path.exists(model_file):
        print("[detector] Model file not found:", model_file)
        return

    _stop_flag = False
    _running_detector = True
    model_file_ref = {'file': model_file}

    detector_thread = threading.Thread(
        target=_detector_loop, args=(model_file_ref, actionmode), daemon=True
    )
    detector_thread.start()
    print("[detector] Started.")


# -------------------- Real-time Learning Loop --------------------
def _realtime_learning_loop(actionmode: int, model_file_ref: dict):
    global _online_buffer, _stop_flag
    import pandas as pd

    print("[learning] Real-time learning loop started.")

    while not _stop_flag:
        sample = _collect_system_sample()
        with _online_lock:
            _online_buffer.append(sample)

        print(
            f"[learning] Sample collected: CPU={sample['cpu']} RSS={sample['rss']}MB Threads={sample['threads']}")

        if len(_online_buffer) >= ONLINE_BUFFER_LIMIT:
            print("[learning] Buffer limit reached → retraining model...")
            with _online_lock:
                df = pd.DataFrame(_online_buffer)
                _online_buffer = []

            result = train_model(df=df, dataset_names=[
                                 "online_buffer"], model_choice="auto")
            best = result["best_model"]
            print(f"[learning] Model updated → {best['model_name']}")

            # Update reference for detector
            model_file_ref['file'] = best['model_file']

        time.sleep(1)


# -------------------- Combined Real-time Function --------------------
def start_realtime_learning_and_detection(actionmode: int):
    global _running_detector, _stop_flag

    if _running_detector:
        print("[ERROR] Real-time detection already running.")
        return

    model_file_ref = {'file': None}

    learning_thread = threading.Thread(
        target=_realtime_learning_loop, args=(actionmode, model_file_ref), daemon=True)
    learning_thread.start()
    print("[learning] Real-time learning started.")

    while model_file_ref['file'] is None and not _stop_flag:
        print("[detector] Waiting for initial model from learning...")
        time.sleep(1)

    _stop_flag = False
    _running_detector = True
    detector_thread = threading.Thread(
        target=_detector_loop, args=(model_file_ref, actionmode), daemon=True)
    detector_thread.start()
    print("[detector] Real-time detection thread started.")

    print("[info] Press Ctrl+C to stop real-time learning and detection...")

    while _running_detector and not _stop_flag:
        time.sleep(1)

    print("[info] Real-time learning and detection stopped.")


# -------------------- Control --------------------
def is_running():
    global _running_detector
    return _running_detector


def stop_all():
    global _stop_flag
    _stop_flag = True
    print("[control] Stopping all real-time processes...")