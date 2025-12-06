from fastapi import FastAPI, Request
import psutil, uuid, asyncio, json, threading
import pandas as pd
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import traceback
import subprocess
import math
import platform

app = FastAPI()

# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MASTER_CSV = DATA_DIR / "all_data.csv"

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SNAPSHOT_INTERVAL = 10       # take process snapshot every 10 sec
TRAIN_INTERVAL = 60          # training every 60 sec (unused now)
AGG_WINDOW_SECONDS = 300     # 5-minute aggregation window (unused now)
TDP_W = float(os.getenv("CPU_TDP_W", "15"))  # nominal CPU TDP for heuristic power (watts)

DEDUP_SUBSET = [
    "Timestamp", "PID", "Process_Name",
    "Power_W", "CPU_Usage_%", "Mem_Usage_MB"
]

MASTER_COLUMNS = DEDUP_SUBSET  # master CSV will only keep these columns

# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------
scanning_active = False
last_training_result = None   # kept for compatibility, but unused
current_run_id = None
current_run_dir = None

loaded_models = []            # unused now
scaler_state = None           # unused now

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
snapshot_task = None

async def snapshot_loop():
    global scanning_active, current_run_dir
    while scanning_active:
        rows = get_process_list_snapshot()
        write_snapshot_files(rows, current_run_dir)
        await asyncio.sleep(SNAPSHOT_INTERVAL)

def safe_float(v):
    try:
        f = float(v)
        if not math.isfinite(f):
            return None
        return f
    except Exception:
        return None

def safe_str(v):
    try:
        import pandas as pd
        if pd.isna(v):
            return None
    except Exception:
        pass
    if v is None:
        return None
    return str(v)

def prime_cpu_percent():
    # Prime the cpu_percent measurement to avoid initial zero
    psutil.cpu_percent(interval=None)

def get_cpu_package_power():
    if platform.system().lower() != "darwin":
        return 0.0
    try:
        result = subprocess.run(
            ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "smc", "-n", "1"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"powermetrics failed (rc={result.returncode}): {result.stderr.strip()}")
            return 0.0
        for line in result.stdout.splitlines():
            if "CPU package power" in line:
                power_str = line.split(":")[1].strip().split(" ")[0]
                return float(power_str)
    except Exception as e:
        print(f"Failed to get CPU power: {e}")
    return 0.0

def estimate_package_power(tdp_w: float = TDP_W) -> float:
    """
    Heuristic package power estimate: TDP * CPU_util * freq_ratio.
    Not a replacement for real sensors, but useful when powermetrics is unavailable.
    """
    try:
        cpu_util = psutil.cpu_percent(interval=None)  # since last call
        freq = psutil.cpu_freq()
        if freq is not None:
            base = freq.max or freq.current or 0.0
            freq_ratio = (freq.current / base) if base > 0 else 1.0
            # Clamp to a sane range
            freq_ratio = max(0.0, min(freq_ratio, 1.5))
        else:
            freq_ratio = 1.0
        return tdp_w * (cpu_util / 100.0) * freq_ratio
    except Exception as e:
        print(f"Failed to estimate package power: {e}")
        return 0.0

# Call this once at startup, or in start_scanning before the loop
prime_cpu_percent()

def get_process_list_snapshot():
    rows = []
    ts = datetime.now(timezone.utc).isoformat()

    cpu_package_power = get_cpu_package_power()
    if cpu_package_power <= 0.0:
        cpu_package_power = estimate_package_power()

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            pid = proc.pid

            # Use a small interval to avoid zero readings
            cpu = proc.cpu_percent(interval=0.1)
            memory = proc.memory_info()
            ram = memory.rss / (1024 * 1024)  # MB

            power_w = cpu_package_power * (cpu / 100.0)

            rows.append({
                "Timestamp": ts,
                "PID": pid,
                "Process_Name": proc.info.get("name") or "Unknown",
                "CPU_Usage_%": float(cpu),
                "Mem_Usage_MB": float(ram),
                "Power_W": float(power_w),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            traceback.print_exc()
            continue

    return rows

def write_snapshot_files(rows, run_dir: Path = None):
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    target_dir = run_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = target_dir / f"snapshot_{now}.csv"
    json_path = target_dir / f"snapshot_{now}.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path

def merge_into_master():
    csv_files = sorted(
        list(DATA_DIR.glob("snapshot_*.csv")) +
        list(Path(DATA_DIR, "runs").glob("**/snapshot_*.csv"))
    )
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)

    # Replace infs and normalize
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Safe PID + Process_Name cleanup
    if "PID" in merged.columns:
        merged["PID"] = pd.to_numeric(merged["PID"], errors="coerce")
        merged["PID"] = merged["PID"].fillna(-1).astype(int)
    if "Process_Name" in merged.columns:
        merged["Process_Name"] = merged["Process_Name"].fillna("Unknown")
    if "Timestamp" in merged.columns:
        merged["Timestamp"] = merged["Timestamp"].fillna("")

    merged_before = len(merged)

    try:
        if set(DEDUP_SUBSET).issubset(merged.columns):
            merged = merged.drop_duplicates(subset=DEDUP_SUBSET)
        else:
            merged = merged.drop_duplicates()
    except Exception:
        merged = merged.drop_duplicates()
    merged_after = len(merged)

    # Keep only the desired feature columns in master
    merged = merged[[c for c in MASTER_COLUMNS if c in merged.columns]]

    merged.to_csv(MASTER_CSV, index=False)

    return {
        "merged_rows_before_dedup": merged_before,
        "merged_rows_after_dedup": merged_after,
        "unique_processes": merged["PID"].astype(str).nunique() if "PID" in merged.columns else 0
    }

# --------------------------------------------------
# API Endpoints (data-only, no training)
# --------------------------------------------------

@app.post("/start-scanning")
async def start_scanning():
    global scanning_active, current_run_id, current_run_dir, snapshot_task
    scanning_active = True
    current_run_id = datetime.now().strftime("run_%Y%m%d%H%M%S")
    current_run_dir = DATA_DIR / "runs" / current_run_id
    current_run_dir.mkdir(parents=True, exist_ok=True)
    # start background snapshotter
    snapshot_task = asyncio.create_task(snapshot_loop())
    print(f"Starting new run: {current_run_id} -> {current_run_dir}")
    return {"status": "scanning started", "scanning": scanning_active, "run_id": current_run_id}

@app.post("/stop-scanning")
async def stop_scanning():
    global scanning_active, snapshot_task
    scanning_active = False
    # wait for snapshot loop to finish
    if snapshot_task:
        await snapshot_task
        snapshot_task = None
    merge_stats = merge_into_master()
    processes = []

    return {
        "status": "scanning stopped",
        "scanning": scanning_active,
        "merge_stats": merge_stats,
        "processes": processes,
    }

@app.get("/scanning-status")
async def get_scanning_status():
    return {
        "scanning": scanning_active,
        "last_training": None,  # training disabled
        "models_loaded": 0      # models unused
    }

@app.post("/predict-anomalies")
async def predict_anomalies_endpoint(request: Request):
    """
    Anomaly prediction disabled for now.
    """
    return {"anomalies": [], "error": "anomaly prediction disabled"}

@app.get("/anomaly-report")
async def get_latest_anomaly_report():
    processes = []
    if MASTER_CSV.exists():
        try:
            df = pd.read_csv(MASTER_CSV).replace([np.inf, -np.inf], np.nan)
            for r in df.to_dict("records"):
                pid_raw = r.get("PID", -1)
                pid = int(pid_raw) if not pd.isna(pid_raw) else -1
                processes.append({
                    "pid": pid,
                    "name": safe_str(r.get("Process_Name")),
                    "cpu": safe_float(r.get("CPU_Usage_%", 0.0)),
                    "ram": safe_float(r.get("Mem_Usage_MB", 0.0)),
                    "power_w": safe_float(r.get("Power_W", 0.0)),
                    "timestamp": safe_str(r.get("Timestamp")),
                })
        except Exception:
            processes = []
    system = {"timestamp": datetime.now().isoformat(), "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}
    return {"anomalies": [], "processes": processes, "models": [], "system": system}

@app.get("/system-stats")
async def system_stats():
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent
    }

@app.get("/processes")
async def get_processes():
    if MASTER_CSV.exists():
        try:
            df = pd.read_csv(MASTER_CSV).replace([np.inf, -np.inf], np.nan)
            processes = []
            for r in df.to_dict("records"):
                pid_raw = r.get("PID", -1)
                pid = int(pid_raw) if not pd.isna(pid_raw) else -1
                processes.append({
                    "pid": pid,
                    "name": safe_str(r.get("Process_Name")),
                    "cpu": safe_float(r.get("CPU_Usage_%", 0.0)),
                    "ram": safe_float(r.get("Mem_Usage_MB", 0.0)),
                    "power_w": safe_float(r.get("Power_W", 0.0)),
                    "timestamp": safe_str(r.get("Timestamp")),
                })
            return {"processes": processes, "source": "master_dedup"}
        except Exception:
            pass

@app.get("/models")
async def list_models():
    # Training disabled; models are not managed
    return {"models": [], "total": 0}

@app.post("/trigger-train")
async def trigger_train():
    """
    Training disabled for now.
    """
    return {"status": "disabled", "result": None}