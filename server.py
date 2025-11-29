from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import psutil, uuid, asyncio, json, joblib, pandas as pd
import os
from datetime import datetime

app = FastAPI()

# Directory where models are stored
MODEL_DIR = "model"

# Load models in memory for quick real-time detection
loaded_models = []

def load_models():
    global loaded_models
    loaded_models = []
    for jf in os.listdir(MODEL_DIR):
        if jf.endswith(".pkl"):
            model_path = os.path.join(MODEL_DIR, jf)
            try:
                pipe = joblib.load(model_path)
                # Find corresponding JSON metadata
                json_file = model_path.replace(".pkl", ".json")
                if os.path.exists(json_file):
                    with open(json_file, "r") as f:
                        meta = json.load(f)
                    loaded_models.append({"pipeline": pipe, "meta": meta})
            except Exception as e:
                print(f"Error loading model {jf}: {e}")

load_models()

# ----------------------------
# Utilities
# ----------------------------
def get_process_list():
    result = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            result.append({
                "id": str(uuid.uuid4()),
                "pid": proc.info['pid'],
                "name": proc.info.get("name", "Unknown"),
                "cpu": proc.info["cpu_percent"],
                "ram": proc.info["memory_info"].rss / (1024 * 1024)
            })
        except:
            continue
    return result

def detect_anomalies(processes):
    """
    Run all loaded models on the process data.
    Returns a list of anomalies detected.
    """
    anomalies = []
    if not loaded_models:
        return anomalies

    # Convert processes to DataFrame
    df = pd.DataFrame(processes)
    if df.empty:
        return anomalies

    # For each model, predict anomalies
    for model in loaded_models:
        pipe = model["pipeline"]
        meta = model["meta"]
        try:
            # Ensure columns are correct
            X = df[['cpu', 'ram']].to_numpy()
            preds = pipe.predict(X)
            # For IsolationForest/DBSCAN: -1 is anomaly
            for i, p in enumerate(preds):
                if p == -1:
                    anomalies.append({
                        "process": df.iloc[i]['name'],
                        "pid": int(df.iloc[i]['pid']),
                        "model": meta["model_name"],
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"Error running model {meta['model_name']}: {e}")

    return anomalies

# ----------------------------
# SSE streaming endpoint
# ----------------------------
async def stream_data():
    while True:
        # System stats
        system_stats = {
            "type": "system",
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "timestamp": datetime.now().isoformat()
        }
        yield {"data": json.dumps(system_stats)}

        # Processes
        processes = {
            "type": "processes",
            "processes": get_process_list(),
            "timestamp": datetime.now().isoformat()
        }
        yield {"data": json.dumps(processes)}

        # Anomalies
        anomalies = {
            "type": "anomalies",
            "anomalies": detect_anomalies(processes["processes"]),
            "timestamp": datetime.now().isoformat()
        }
        yield {"data": json.dumps(anomalies)}

        await asyncio.sleep(1)

@app.get("/stream")
async def stream():
    return EventSourceResponse(stream_data())
