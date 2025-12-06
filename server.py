from fastapi import FastAPI, Request
import psutil, uuid, asyncio, json, joblib, pandas as pd
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import traceback

from src.models. DBSA import train_with_dbscan
from src.models.IsolationForest import train_with_isolation_forest
from src.models. AdaBoost import train_adaboost_with_iso_dbscan, partial_train_on_batch

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
current_run_id = None
current_run_dir = None

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
    # include top-level and run subfolder snapshots
    csv_files = sorted(list(DATA_DIR.glob("snapshot_*.csv")) + list(Path(DATA_DIR, "runs").glob("**/snapshot_*.csv")))
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
        merged["pid"] = pd.to_numeric(merged["pid"], errors="coerce").fillna(-1).astype(int)
    merged_before = len(merged)
    # Prefer targeted dedupe subset when available, otherwise drop exact duplicates
    try:
        if set(DEDUP_SUBSET).issubset(merged.columns):
            merged = merged.drop_duplicates(subset=DEDUP_SUBSET)
        else:
            merged = merged.drop_duplicates()
    except Exception:
        merged = merged.drop_duplicates()
    merged_after = len(merged)

    # limit master size to avoid unbalanced huge datasets (keep latest 50k rows)
    MAX_MASTER_ROWS = 50000
    if len(merged) > MAX_MASTER_ROWS:
        merged = merged.sort_values(by="timestamp", ascending=False).head(MAX_MASTER_ROWS)

    merged["process_key"] = merged["pid"].astype(str) + "||" + merged["name"].astype(str)
    occ = merged.groupby("process_key").size().rename("occurrence_count")
    merged = merged.merge(occ, how="left", left_on="process_key", right_index=True)
    merged["timestamp_parsed"] = pd.to_datetime(merged.get("timestamp"), utc=True, errors="coerce")
    last_seen = merged.groupby("process_key")["timestamp_parsed"].max().rename("last_seen")
    merged = merged.merge(last_seen, how="left", left_on="process_key", right_index=True)
    merged.to_csv(MASTER_CSV, index=False)
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
        _, X = build_aggregated_features(use_window_seconds=AGG_WINDOW_SECONDS)
        if X.shape[0] == 0:
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
        ada_meta = train_adaboost_with_iso_dbscan(X_scaled, iso_preds_local, db_preds_local, str(MODEL_DIR_PATH))

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
    
    # look for top-level and per-run models
    pkl_files = list(MODEL_DIR.glob("*.pkl"))
    # include run subfolders
    pkl_files += [p for p in MODEL_DIR.glob("runs/**") if p.suffix == ".pkl"]
    print(f"🔍 Found {len(pkl_files)} . pkl files (including run subfolders)")
    
    for file_path in pkl_files:
        print(f"  - {file_path.name}")
        try:
            loaded_obj = joblib.load(file_path)
            meta = {}
            json_path = file_path.with_suffix(".json")

            if json_path.exists():
                with open(json_path, "r") as f:
                    meta = json.load(f)
                print(f"  ✅ Loaded {meta.get('model_name', 'unknown')}")
            else:
                print(f"  ⚠️  No metadata file: {json_path.name}")

            # If the saved artifact is a dict produced by our online trainer, extract model
            if isinstance(loaded_obj, dict) and "model" in loaded_obj:
                pipe = loaded_obj["model"]
                # annotate meta if missing
                if not meta:
                    meta = {"model_name": file_path.stem, "model_type": "AdaBoost_online_proxy", "model_file": str(file_path)}
            else:
                pipe = loaded_obj

            mtype = meta.get("model_type", "UNKNOWN")
            loaded_models.append({
                "model_type": mtype,
                "pipeline": pipe,
                "meta": meta
            })
        except Exception as e:
            print(f"  ❌ Failed to load {file_path.name}: {e}")
    
    print(f"✅ Total models loaded: {len(loaded_models)}")


# Helper to dedupe anomalies by process+pid keeping latest timestamp

def dedupe_anomalies_by_process(anomalies):
    if not anomalies:
        return anomalies
    # convert timestamps to datetime if present
    for a in anomalies:
        try:
            a['_ts_parsed'] = pd.to_datetime(a.get('timestamp'))
        except Exception:
            a['_ts_parsed'] = pd.Timestamp(0)
    # build map of best (latest) anomaly per process+pid
    best = {}
    for a in anomalies:
        key = f"{a.get('process')}_{a.get('pid')}"
        cur = best.get(key)
        if cur is None or a['_ts_parsed'] > cur['_ts_parsed']:
            best[key] = a
    # strip helper field
    res = []
    for v in best.values():
        v.pop('_ts_parsed', None)
        res.append(v)
    return res

def sanitize_for_json(data):
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    if isinstance(data, tuple):
        return [sanitize_for_json(item) for item in data]
    if isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    if isinstance(data, pd.Timestamp):
        return data.isoformat()
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, (np.floating, float)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    try:
        if pd.isna(data):
            return None
    except Exception:
        pass
    return data

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
            # Convert features to numpy array (strip column names) and apply scaler if present
            try:
                raw_X = X.values if hasattr(X, 'values') else np.asarray(X)
            except Exception:
                raw_X = np.asarray(X)

            # If pipeline has a scaler that expects feature names, try to use DataFrame to align columns
            scaler = None
            if hasattr(pipe, 'named_steps') and 'scaler' in pipe.named_steps:
                scaler = pipe.named_steps['scaler']

            if scaler is not None and hasattr(scaler, 'feature_names_in_') and hasattr(X, 'columns'):
                # Align columns to what scaler expects
                try:
                    aligned = X.reindex(columns=scaler.feature_names_in_).fillna(0.0)
                    scaled = scaler.transform(aligned)
                except Exception:
                    scaled = scaler.transform(raw_X)
            elif scaler is not None:
                try:
                    scaled = scaler.transform(raw_X)
                except Exception:
                    scaled = raw_X
            else:
                scaled = raw_X

            # Get final estimator (last step) if pipeline, else use pipe directly
            final_estimator = pipe
            try:
                if hasattr(pipe, 'steps') and len(pipe.steps) > 0:
                    final_estimator = pipe.steps[-1][1]
            except Exception:
                final_estimator = pipe

            # Ensure numpy array input for final predict
            input_for_pred = scaled if isinstance(scaled, np.ndarray) else np.asarray(scaled)

            preds = final_estimator.predict(input_for_pred)

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
                csv_path, json_path = write_snapshot_files(rows, run_dir=current_run_dir)
                csv_path, _ = write_snapshot_files(rows, run_dir=current_run_dir)
                print(f"Saved snapshot: {csv_path}")
                # quick incremental training per snapshot: build features and call partial_fit with small batch
                try:
                    # aggregate current snapshot into features
                    import pandas as _pd
                    df_snapshot = _pd.read_csv(csv_path)
                    # Build per-process aggregated features for this snapshot
                    process_groups = df_snapshot.groupby(["pid", "name"]).agg(
                        occurrence_count=("pid", "size"),
                        mean_cpu=("cpu", "mean"),
                        max_cpu=("cpu", "max"),
                        std_cpu=("cpu", "std"),
                        mean_ram=("ram", "mean"),
                        max_ram=("ram", "max"),
                        std_ram=("ram", "std")
                    ).reset_index()
                    for c in ["std_cpu", "std_ram"]:
                        if c in process_groups.columns:
                            process_groups[c] = process_groups[c].fillna(0.0)
                    # construct feature matrix
                    feat_cols = ["mean_cpu", "max_cpu", "std_cpu", "mean_ram", "max_ram", "std_ram", "occurrence_count"]
                    # add last_seen_age_seconds as 0 for snapshot
                    process_groups["last_seen_age_seconds"] = 0.0
                    # labels: use isolation forest + dbscan heuristics from most recent full models if available
                    # fallback: label everything 0 (safe)
                    y_batch = [0] * X_batch.shape[0]
                    try:
                        # if models exist, run them on X_batch
                        # load latest DBSCAN and ISO models from disk if present
                        iso_files = list(MODEL_DIR.glob("runs/**/ISO_*.pkl")) + list(MODEL_DIR.glob("ISO_*.pkl"))
                        db_files = list(MODEL_DIR.glob("runs/**/DBSCAN_*.pkl")) + list(MODEL_DIR.glob("DBSCAN_*.pkl"))
                        if iso_files:
                            iso_pipe = joblib.load(max(iso_files, key=os.path.getmtime))
                            iso_preds = iso_pipe.predict(X_batch)
                        else:
                            iso_preds = [1] * X_batch.shape[0]
                        if db_files:
                            db_pipe = joblib.load(max(db_files, key=os.path.getmtime))
                            # if dbscan pipeline, access dbscan.labels_ after fit_predict; use predict-like logic
                            try:
                                db_labels = db_pipe.named_steps['dbscan'].fit_predict(X_batch)
                            except Exception:
                                db_labels = db_pipe.fit_predict(X_batch)
                        else:
                            db_labels = [0] * X_batch.shape[0]

                        db_anomaly = np.array([1 if x == -1 else 0 for x in db_labels])
                        iso_anomaly = np.array([1 if x == -1 else 0 for x in iso_preds])
                        y_batch = np.where((db_anomaly == 1) | (iso_anomaly == 1), 1, 0).astype(int)
                    except Exception:
                        y_batch = [0] * X_batch.shape[0]

                    # scale batch using scaler_state if available
                    try:
                        if scaler_state is not None:
                            try:
                                # If scaler knows feature names, align DataFrame columns to avoid warnings
                                if hasattr(scaler_state, 'feature_names_in_'):
                                    import pandas as _pd
                                    cols = list(scaler_state.feature_names_in_)
                                    df_batch = _pd.DataFrame(X_batch, columns=["mean_cpu", "max_cpu", "std_cpu", "mean_ram", "max_ram", "std_ram", "occurrence_count", "last_seen_age_seconds"]) 
                                    aligned = df_batch.reindex(columns=cols).fillna(0.0)
                                    Xb_scaled = scaler_state.transform(aligned)
                                else:
                                    Xb_scaled = scaler_state.transform(X_batch)
                            except Exception:
                                Xb_scaled = scaler_state.transform(X_batch)
                        else:
                            Xb_scaled = X_batch
                        # call incremental trainer
                        partial_train_on_batch(Xb_scaled, y_batch, str(MODEL_DIR))
                    except Exception as e:
                        print("Incremental train error:", e)
                except Exception as e:
                    print("Snapshot incremental training skipped:", e)

                now_ts = asyncio.get_event_loop(). time()
                if (now_ts - last_train) >= TRAIN_INTERVAL:
                    print("Merging and training models...")
                    # create per-run model dir
                    run_model_dir = MODEL_DIR / "runs" / (current_run_id or "run_unknown")
                    run_model_dir.mkdir(parents=True, exist_ok=True)
                    result = train_all_models(MODEL_DIR_PATH=run_model_dir)
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
    global scanning_active, current_run_id, current_run_dir
    scanning_active = True
    # generate a run id and directory
    current_run_id = datetime.now().strftime("run_%Y%m%d%H%M%S")
    current_run_dir = DATA_DIR / "runs" / current_run_id
    current_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting new run: {current_run_id} -> {current_run_dir}")
    return {"status": "scanning started", "scanning": scanning_active, "run_id": current_run_id}

@app.post("/stop-scanning")
async def stop_scanning():
    """Stop collecting new data and return anomaly report, processes and model list plus system stats"""
    global scanning_active
    scanning_active = False

    merge_stats = merge_into_master()

    processes = []
    agg_df, _ = build_aggregated_features(use_window_seconds=AGG_WINDOW_SECONDS)
    if not agg_df.empty:
        for _, r in agg_df.iterrows():
            processes.append({
                "pid": int(r.get("pid", -1)),
                "name": r.get("name"),
                "cpu": float(r.get("mean_cpu", 0.0)),
                "ram": float(r.get("mean_ram", 0.0)),
                "last_seen": r.get("last_seen")
            })
    else:
        csvs = sorted(DATA_DIR.glob("snapshot_*.csv"))
        if csvs:
            try:
                df = pd.read_csv(csvs[-1])
                processes = df.to_dict("records")
            except Exception:
                processes = []

    anomalies = detect_anomalies_from_loaded_models(processes)
    anomalies = dedupe_anomalies_by_process(anomalies)

    processes = sanitize_for_json(processes)
    anomalies = sanitize_for_json(anomalies)
    models_meta = [m.get("meta", {}) for m in loaded_models]
    system = {"timestamp": datetime.now().isoformat(), "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}

    return {
        "status": "scanning stopped",
        "scanning": scanning_active,
        "last_training": last_training_result,
        "merge_stats": merge_stats,
        "anomalies": anomalies,
        "processes": processes,
        "models": models_meta,
        "system": system
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
        results = dedupe_anomalies_by_process(results)
        return {"anomalies": results, "error": None}
    except Exception as e:
        print(f"Error in predict-anomalies: {e}")
        traceback.print_exc()
        return {"anomalies": [], "error": str(e)}

@app.get("/anomaly-report")
async def get_latest_anomaly_report():
    """Get anomalies and processes from latest aggregated data along with models and system stats"""
    # prefer master aggregated data
    if MASTER_CSV.exists():
        try:
            df = pd.read_csv(MASTER_CSV)
            processes = []
            for r in df.to_dict("records"):
                processes.append({
                    "pid": int(r.get("pid", -1)) if "pid" in r else -1,
                    "name": r.get("name"),
                    "cpu": float(r.get("mean_cpu", 0.0)) if "mean_cpu" in r else None,
                    "ram": float(r.get("mean_ram", 0.0)) if "mean_ram" in r else None,
                    "last_seen": r.get("last_seen")
                })
        except Exception:
            processes = []
    else:
        csvs = sorted(DATA_DIR.glob("snapshot_*.csv"))
        if not csvs:
            return {"anomalies": [], "processes": [], "models": [], "system": {"timestamp": datetime.now().isoformat(), "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}}
        latest_csv = csvs[-1]
        df = pd.read_csv(latest_csv)
        processes = df.to_dict("records")

    anomalies = detect_anomalies_from_loaded_models(processes)
    anomalies = dedupe_anomalies_by_process(anomalies)
    anomalies = detect_anomalies_from_loaded_models(processes)
    anomalies = dedupe_anomalies_by_process(anomalies)
    anomalies = sanitize_for_json(anomalies)
    processes = sanitize_for_json(processes)
    models_meta = [m.get("meta", {}) for m in loaded_models]
    system = {"timestamp": datetime.now().isoformat(), "cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}

    return {"anomalies": anomalies, "processes": processes, "models": models_meta, "system": system}
async def system_stats():
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory(). percent
    }

@app.get("/processes")
async def get_processes():
    now = datetime.now().isoformat()
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            cpu = proc.info.get("cpu_percent", None) or proc.cpu_percent(interval=None)
            ram = proc.info["memory_info"].rss / (1024 * 1024) if proc.info.get("memory_info") else None
            proc_record = {
                "id": str(uuid.uuid4()),
                "timestamp": now,
                "pid": int(proc.info.get("pid", -1)),
                "name": proc.info.get("name") or "Unknown",
                "cpu": float(cpu) if cpu is not None else None,
                "ram": float(ram) if ram is not None else None
            }
            processes.append(proc_record)
        except Exception as e:
            # log individual process errors
            print(f"process iterate error: {e}")
            continue
    # debug log sample
    if processes:
        try:
            print(f"/processes sample: pid={processes[0]['pid']}, cpu={processes[0]['cpu']}, ram={processes[0]['ram']}")
        except Exception:
            pass
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