import streamlit as st
import pandas as pd
import json
import glob
import threading
from src.runner import (
    run_detection_with_saved_model,
    run_realtime_learning_step,
    stop_detector,
    is_running
)
from src.train import train_model
from src.data.collect import collect_and_save

st.set_page_config(page_title="System Anomaly Detection", layout="wide")
st.title("System Anomaly Detection 🚀")
st.markdown("Modern interface for anomaly detection system")

# ---------------- Sidebar ----------------
option = st.sidebar.selectbox(
    "Menu",
    ["Manual", "Stop Background Detection",
     "Detect Saved Model", "Real-time Learning"]
)

# ---------------- Manual Page ----------------
if option == "Manual":
    st.header("Manual Operations")
    tabs = st.tabs(["Data Collection", "Train Model"])

    # --------- Tab 1: Data Collection ---------
    with tabs[0]:
        st.subheader("Collect System Data")
        duration = st.number_input(
            "Duration (seconds)", min_value=1, max_value=3600, value=15)

        if st.button("Start Data Collection"):
            collect_and_save(duration=duration, interval=1)
            st.success(f"Data collection completed ({duration} seconds).")

    # --------- Tab 2: Train Model ---------
    with tabs[1]:
        st.subheader("Train New Model")

        uploaded_files = st.file_uploader(
            "Upload CSV file(s)", accept_multiple_files=True, type=["csv"]
        )

        model_choice = st.radio(
            "Select Model Type",
            ["auto", "isolation", "dbscan"],
            help="Auto selects the best performing model."
        )

        if st.button("Train Model Now"):
            if not uploaded_files:
                st.warning("Upload one or more CSV files first.")
            else:
                dfs = [pd.read_csv(f) for f in uploaded_files]
                df = pd.concat(dfs, ignore_index=True)
                dataset_names = [f.name for f in uploaded_files]

                result = train_model(
                    df=df,
                    dataset_names=dataset_names,
                    model_choice=model_choice
                )

                best = result["best_model"]

                st.success(f"Model trained successfully: {best['model_name']}")
                st.json(best)


# ---------------- Stop Background Detection ----------------
elif option == "Stop Background Detection":
    st.header("Stop Background Detector")

    if is_running():
        stop_detector()
        st.success("Detector stopped.")
    else:
        st.info("Detector is not running.")


# ---------------- Detect Saved Model ----------------
elif option == "Detect Saved Model":
    st.header("Use Saved Model for Live Detection")

    json_files = glob.glob("model/*.json")

    if not json_files:
        st.warning("No saved models found.")
    else:
        models = []
        for jf in json_files:
            with open(jf, "r") as f:
                models.append(json.load(f))

        model_names = [m["model_name"] for m in models]
        selected_name = st.selectbox("Select a model", model_names)
        model_meta = next(
            m for m in models if m["model_name"] == selected_name)

        # Display metadata
        st.subheader("Model Metadata")
        st.json(model_meta)

        if st.button("Start Detection"):
            thread = threading.Thread(
                target=run_detection_with_saved_model,
                args=(model_meta["model_file"],),
                daemon=True
            )
            thread.start()
            st.success("Detection started (check terminal logs).")


# ---------------- Real-time Learning ----------------
elif option == "Real-time Learning":
    st.header("Real-time Learning & Background Anomaly Detection")

    # Init session state
    if "learning_thread" not in st.session_state:
        st.session_state.learning_thread = None
    if "log" not in st.session_state:
        st.session_state.log = []
    if "running" not in st.session_state:
        st.session_state.running = False

    log_box = st.empty()

    # Background learning loop
    def learning_loop():
        while st.session_state.running:
            result = run_realtime_learning_step()

            if "sample" in result:
                st.session_state.log.append(
                    {"type": "sample", **result["sample"]}
                )

            if "updated_model" in result:
                st.session_state.log.append(
                    {"type": "model_update", **result["updated_model"]}
                )

            df = pd.DataFrame(st.session_state.log)
            log_box.dataframe(df)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Start Real-time Learning", disabled=st.session_state.running):
            st.session_state.running = True
            st.session_state.learning_thread = threading.Thread(
                target=learning_loop,
                daemon=True
            )
            st.session_state.learning_thread.start()
            st.success("Real-time learning started.")

    with col2:
        if st.button("Stop Real-time Learning"):
            st.session_state.running = False
            st.success("Real-time learning stopped.")

    if st.session_state.log:
        log_box.dataframe(pd.DataFrame(st.session_state.log))