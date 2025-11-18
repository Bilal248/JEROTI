import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
 
MODEL_PATH = "../model/saved_model.pkl"
DATA_PATH = "../data/sample_training_data.csv"   

def train_model():
    """
    Train IsolationForest on process metrics (cpu, rss, threads)
    and save model + scaler as a pickle file.
    """
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Features for training
    X = df[["cpu", "rss", "threads"]]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    # Isolation Forest (often abbreviated as iForest) is a machine learning algorithm 
    # for anomaly or outlier detection. Unlike many other algorithms that profile “normal” data, 
    # Isolation Forest explicitly isolates anomalies. Here’s a detailed breakdown:

    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42
    )
    model.fit(X_scaled)

    # Save model and scaler together
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"Model trained and saved at: {MODEL_PATH}")