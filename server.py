from fastapi import FastAPI, Request
from sse_starlette. sse import EventSourceResponse
import psutil, uuid, asyncio, json, joblib, pandas as pd
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import traceback

from src.models. DBSA import train_with_dbscan
from src.models.IsolationForest import train_with_isolation_forest
from src.models. AdaBoost import train_adaboost_with_iso_dbscan

app = FastAPI()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MASTER_CSV = DATA_DIR / "all_data.csv"
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

SNAPSHOT_INTERVAL = 10
TRAIN_INTERVAL = 60
AGG_WINDOW_SECONDS = None
DEDUP_SUBSET = ["pid", "name", "cpu", "ram", "timestamp"]

# Global state for scanning control
scanning_active = False
last_training_result = None

# Loaded models for predictions in memory
loaded_models = []
scaler_state = None  # Store scaler for consistent feature transformation

# Feature columns used during training
REQUIRED_FEATURES = ["mean_cpu", "max_cpu", "std_cpu", "mean_ram", "max_ram", "std_ram", "occurrence_count", "last_seen_age_seconds"]

# -------------------------------------
# Utilities for snapshot/aggregation
# -------------------------------------
def get_process_list_snapshot():
    rows = []
    ts = datetime.now(timezone.utc). isoformat()
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            cpu = proc.info. get("cpu_percent", None) or proc.cpu_percent(interval=None)
            ram = proc.info["memory_info"].rss / (1024 * 1024) if proc.info. get("memory_info") else None
            rows.append({
                "id": str(uuid.uuid4()),
                "timestamp": ts,
                "pid": int(proc.info.get("pid", -1)),
                "name": proc.info.get("name") or "Unknown",
                "cpu": float(cpu) if cpu is not None else None,
                "ram": float(ram) if ram is not None else None
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        except Exception:
            continue
    return rows

def write_snapshot_files(rows):
    now = datetime.now(). strftime("%Y%m%d%H%M%S")
    csv_path = DATA_DIR / f"snapshot_{now}.csv"
    json_path = DATA_DIR / f"snapshot_{now}.json"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json. dump(rows, f, indent=2)
    return csv_path, json_path

def merge_into_master():
    csv_files = sorted(DATA_DIR.glob("snapshot_*.csv"))
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
    if "pid" in merged.columns:
        merged["pid"] = merged["pid"].astype(int)
    merged_before = len(merged)
    if set(DEDUP_SUBSET). issubset(merged.columns):
        merged = merged.drop_duplicates(subset=DEDUP_SUBSET)
    else:
        merged = merged.drop_duplicates()
    merged_after = len(merged)
    merged["process_key"] = merged["pid"].astype(str) + "||" + merged["name"].astype(str)
    occ = merged.groupby("process_key").size(). rename("occurrence_count")
    merged = merged.merge(occ, how="left", left_on="process_key", right_index=True)
    merged["timestamp_parsed"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    last_seen = merged.groupby("process_key")["timestamp_parsed"].max(). rename("last_seen")
    merged = merged.merge(last_seen, how="left", left_on="process_key", right_index=True)
    merged. to_csv(MASTER_CSV, index=False)
    return {
        "merged_rows_before_dedup": merged_before,
        "merged_rows_after_dedup": merged_after,
        "unique_processes": merged["process_key"].nunique()
    }

def build_aggregated_features(use_window_seconds=None):
    if not MASTER_CSV.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(MASTER_CSV, parse_dates=["timestamp"])
    if use_window_seconds is not None:
        cutoff = pd. Timestamp.utcnow() - pd.to_timedelta(use_window_seconds, unit="s")
        df = df[df["timestamp"] >= cutoff]
    df["cpu"] = pd.to_numeric(df["cpu"], errors="coerce").fillna(0.0)
    df["ram"] = pd.to_numeric(df["ram"], errors="coerce").fillna(0.0)
    df["process_key"] = df["pid"]. astype(str) + "||" + df["name"].astype(str)
    agg = df.groupby("process_key").agg(
        pid = ("pid", "first"),
        name = ("name", "first"),
        occurrence_count = ("process_key", "size"),
        mean_cpu = ("cpu", "mean"),
        max_cpu = ("cpu", "max"),
        std_cpu = ("cpu", "std"),
        mean_ram = ("ram", "mean"),
        max_ram = ("ram", "max"),
        std_ram = ("ram", "std"),
        last_seen = ("timestamp", "max"),
    ).reset_index(drop=True)
    for c in ["std_cpu", "std_ram"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)
    agg["last_seen_parsed"] = pd.to_datetime(agg["last_seen"], utc=True, errors="coerce")
    agg["last_seen_age_seconds"] = (pd.Timestamp.utcnow() - agg["last_seen_parsed"]).dt.total_seconds(). fillna(0.0)
    features = agg[["mean_cpu", "max_cpu", "std_cpu", "mean_ram", "max_ram", "std_ram", "occurrence_count", "last_seen_age_seconds"]].fillna(0.0)
    return agg, features

# -------------------------------------
# Model training and loading
# -------------------------------------
def train_all_models(MODEL_DIR_PATH=MODEL_DIR):
    global scaler_state, last_training_result
    try:
        merge_stats = merge_into_master()
        if merge_stats is None:
            print("No snapshots to merge.")
            return None
        agg_df, X = build_aggregated_features(use_window_seconds=AGG_WINDOW_SECONDS)
        if X.shape[0] == 0:
            print("No aggregated data to train on.")
            return None
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PERSIST SCALER for later inference
        scaler_state = scaler

        feature_stats = {col: {
            "mean": float(X[col].mean()) if col in X.columns else 0.0,
            "std": float(X[col].std()) if col in X.columns else 0.0
        } for col in X.columns}
        dataset_names = [str(p. name) for p in DATA_DIR.glob("snapshot_*.csv")]

        # --- DBSCAN Training
        print("Training DBSCAN...")
        db_meta = train_with_dbscan(X_scaled, feature_stats, dataset_names, str(MODEL_DIR_PATH))
        db_model_file = db_meta["model_file"]
        db_pipe = joblib.load(db_model_file)
        db_preds_local = db_pipe.named_steps["dbscan"].labels_

        # --- Isolation Forest Training
        print("Training IsolationForest...")
        iso_meta = train_with_isolation_forest(X, feature_stats, dataset_names, str(MODEL_DIR_PATH))
        iso_model_file = iso_meta["model_file"]
        iso_pipe = joblib.load(iso_model_file)
        iso_preds_local = iso_pipe.predict(X)

        # --- AdaBoost Training
        print("Training AdaBoost...")
        ada_meta = train_adaboost_with_iso_dbscan(X. values, iso_preds_local, db_preds_local, str(MODEL_DIR_PATH))

        # Reload models into memory
        print("Reloading models into memory...")
        load_models_from_disk()

        result = {
            "db_meta": db_meta,
            "iso_meta": iso_meta,
            "ada_meta": ada_meta,
            "merge_stats": merge_stats,
            "agg_rows": X.shape[0]
        }
        last_training_result = result
        return result
    except Exception as e:
        print("Training error:", e)
        traceback.print_exc()
        return None

def load_models_from_disk():
    global loaded_models
    loaded_models = []
    
    print(f"📁 Looking for models in: {MODEL_DIR}")
    print(f"📁 Model directory exists: {MODEL_DIR.exists()}")
    
    if not MODEL_DIR. exists():
        print("⚠️  Model directory does not exist!")
        return
    
    pkl_files = list(MODEL_DIR.glob("*.pkl"))
    print(f"🔍 Found {len(pkl_files)} . pkl files")
    
    for file_path in pkl_files:
        print(f"  - {file_path.name}")
        try:
            pipe = joblib.load(file_path)
            meta = {}
            json_path = file_path.with_suffix(".json")
            
            if json_path.exists():
                with open(json_path, "r") as f:
                    meta = json.load(f)
                print(f"  ✅ Loaded {meta.get('model_name', 'unknown')}")
            else:
                print(f"  ⚠️  No metadata file: {json_path. name}")
            
            mtype = meta.get("model_type", "UNKNOWN")
            loaded_models.append({
                "model_type": mtype,
                "pipeline": pipe,
                "meta": meta
            })
        except Exception as e:
            print(f"  ❌ Failed to load {file_path.name}: {e}")
    
    print(f"✅ Total models loaded: {len(loaded_models)}")

# ----- FIX: Properly transform input features to 8 features -----
def build_inference_features_from_processes(processes):
    """
    Convert raw process snapshot to aggregated features matching training data. 
    Returns: DataFrame with 8 features needed by models
    """
    if not processes:
        return None
    
    df = pd.DataFrame(processes)
    if df.empty:
        return None
    
    # Ensure numeric types
    df["cpu"] = pd.to_numeric(df. get("cpu", [0] * len(df)), errors="coerce").fillna(0.0)
    df["ram"] = pd.to_numeric(df.get("ram", [0] * len(df)), errors="coerce").fillna(0.0)
    df["pid"] = df. get("pid", [0] * len(df))
    df["name"] = df.get("name", ["Unknown"] * len(df))
    
    df["process_key"] = df["pid"]. astype(str) + "||" + df["name"].astype(str)
    
    agg = df.groupby("process_key").agg(
        pid=("pid", "first"),
        name=("name", "first"),
        occurrence_count=("process_key", "size"),
        mean_cpu=("cpu", "mean"),
        max_cpu=("cpu", "max"),
        std_cpu=("cpu", "std"),
        mean_ram=("ram", "mean"),
        max_ram=("ram", "max"),
        std_ram=("ram", "std"),
    ). reset_index(drop=True)
    
    for c in ["std_cpu", "std_ram"]:
        if c in agg.columns:
            agg[c] = agg[c].fillna(0.0)
    
    # Add last_seen_age_seconds as 0 since this is fresh data
    agg["last_seen_age_seconds"] = 0.0
    
    features = agg[REQUIRED_FEATURES].fillna(0.0)
    return agg, features

# ----- FIX: Use scaler for consistent inference -----
def detect_anomalies_from_loaded_models(processes):
    """
    Run inference on processes using all loaded models. 
    Properly transforms input features to match training data dimensions.
    """
    if not loaded_models:
        print("No models loaded for inference")
        return []
    
    if not processes:
        return []
    
    result = build_inference_features_from_processes(processes)
    if result is None:
        return []
    
    agg_df, X = result
    
    if X.empty or X.shape[0] == 0:
        return []
    
    anomalies = []
    
    for model_info in loaded_models:
        model_type = model_info["model_type"]
        pipe = model_info["pipeline"]
        meta = model_info. get("meta", {})
        
        # Skip DBSCAN for inference
        if model_type. startswith("DBSCAN"):
            continue
        
        try:
            # Use the pipeline's scaler to transform features
            if hasattr(pipe, 'named_steps') and 'scaler' in pipe.named_steps:
                X_transformed = pipe.named_steps["scaler"].transform(X)
            else:
                X_transformed = X
            
            preds = pipe.predict(X)
            
            if "IsolationForest" in model_type:
                for i, p in enumerate(preds):
                    if p == -1:
                        anomalies.append({
                            "process": str(agg_df.iloc[i]["name"]),
                            "pid": int(agg_df.iloc[i]["pid"]),
                            "model": meta.get("model_name", "unknown"),
                            "type": "IsolationForest",
                            "timestamp": datetime.now().isoformat(),
                            "mean_cpu": float(agg_df.iloc[i]["mean_cpu"]),
                            "mean_ram": float(agg_df.iloc[i]["mean_ram"]),
                        })
            
            if "AdaBoost" in model_type or "ADA" in model_type:
                for i, p in enumerate(preds):
                    if int(p) == 1:
                        anomalies.append({
                            "process": str(agg_df. iloc[i]["name"]),
                            "pid": int(agg_df.iloc[i]["pid"]),
                            "model": meta.get("model_name", "unknown"),
                            "type": "AdaBoost",
                            "timestamp": datetime. now().isoformat(),
                            "mean_cpu": float(agg_df.iloc[i]["mean_cpu"]),
                            "mean_ram": float(agg_df.iloc[i]["mean_ram"]),
                        })
        
        except Exception as e:
            print(f"Error running model {meta.get('model_name','? ')}: {e}")
            traceback.print_exc()
    
    return anomalies

# -------------------------------------
# Async BG snapshot + training
# -------------------------------------
async def background_collector():
    global scanning_active
    last_train = 0
    print("Background collector started...")
    while True:
        try:
            if scanning_active:
                rows = get_process_list_snapshot()
                csv_path, json_path = write_snapshot_files(rows)
                print(f"Saved snapshot: {csv_path}")
                
                now_ts = asyncio.get_event_loop(). time()
                if (now_ts - last_train) >= TRAIN_INTERVAL:
                    print("Merging and training models...")
                    result = train_all_models()
                    print("Training result:", result)
                    last_train = now_ts
        except Exception as e:
            print("Background collector error:", e)
            traceback.print_exc()
        
        await asyncio.sleep(SNAPSHOT_INTERVAL)

@app.on_event("startup")
async def startup_event():
    load_models_from_disk()
    asyncio.create_task(background_collector())

# -------------------------------------
# API Endpoints
# -------------------------------------

@app.post("/start-scanning")
async def start_scanning():
    """Start collecting process snapshots and training models"""
    global scanning_active
    scanning_active = True
    return {"status": "scanning started", "scanning": scanning_active}

@app.post("/stop-scanning")
async def stop_scanning():
    """Stop collecting new data"""
    global scanning_active
    scanning_active = False
    return {
        "status": "scanning stopped",
        "scanning": scanning_active,
        "last_training": last_training_result
    }

@app. get("/scanning-status")
async def get_scanning_status():
    """Check current scanning status"""
    return {
        "scanning": scanning_active,
        "last_training": last_training_result,
        "models_loaded": len(loaded_models)
    }

@app. post("/predict-anomalies")
async def predict_anomalies_endpoint(request: Request):
    """
    Predict anomalies from a list of processes. 
    Automatically aggregates process data to match training features.
    """
    try:
        processes = await request.json()
        results = detect_anomalies_from_loaded_models(processes)
        return {"anomalies": results, "error": None}
    except Exception as e:
        print(f"Error in predict-anomalies: {e}")
        traceback.print_exc()
        return {"anomalies": [], "error": str(e)}

@app.get("/anomaly-report")
async def get_latest_anomaly_report():
    """Get anomalies from latest snapshot"""
    csvs = sorted(DATA_DIR.glob("snapshot_*. csv"))
    if not csvs:
        return {"anomalies": []}
    latest_csv = csvs[-1]
    df = pd.read_csv(latest_csv)
    processes = df. to_dict("records")
    anomalies = detect_anomalies_from_loaded_models(processes)
    return {"anomalies": anomalies}

@app.get("/system-stats")
async def system_stats():
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory(). percent
    }

@app.get("/processes")
async def get_processes():
    now = datetime.now(). isoformat()
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            cpu = proc.info. get("cpu_percent", None) or proc.cpu_percent(interval=None)
            ram = proc.info["memory_info"].rss / (1024 * 1024) if proc.info. get("memory_info") else None
            processes.append({
                "id": str(uuid.uuid4()),
                "timestamp": now,
                "pid": int(proc.info. get("pid", -1)),
                "name": proc.info.get("name") or "Unknown",
                "cpu": float(cpu) if cpu is not None else None,
                "ram": float(ram) if ram is not None else None
            })
        except Exception:
            continue
    return {"processes": processes}

@app.get("/models")
async def list_models():
    files = []
    for p in MODEL_DIR.glob("*.json"):
        with open(p, "r") as f:
            files.append(json.load(f))
    return {"models": files, "total": len(files)}

@app.post("/trigger-train")
async def trigger_train():
    """Manually trigger model training"""
    result = train_all_models()
    return {"status": "ok" if result else "failed", "result": result}