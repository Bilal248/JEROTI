# JEROTI

System anomaly detection project (ML) with:

- Data collection via `psutil` (CPU/RSS/threads)
- Offline training (Isolation Forest / DBSCAN / Auto-select)
- Live detection and real-time (online) learning
- CLI interface (recommended) and a Streamlit UI (optional)

## Requirements

- Python 3.12+
- `uv` installed

## Setup

From the project folder:

```bash
uv sync
```

This creates `.venv/` and installs dependencies from `pyproject.toml`.

## Run (CLI)

Start the interactive CLI:

```bash
uv run python ./cli.py
```

### Real-time detection action mode

When starting real-time learning + detection from the CLI, you’ll be asked for an **ActionMode**:

- `1` = try to **kill** the most suspicious anomalous process (by PID)
- `2` = **log only** (no killing)

Logs are written under `logs/`.

## Run (Streamlit UI)

Optional UI:

```bash
uv run streamlit run ./main.py
```

## Training outputs

- Trained models + metadata: `model/` (files like `ISO_*.pkl` / `DBSCAN_*.pkl` and matching `.json`)
- Real-time online model snapshot: `builds/jeroti_online_sgd.pkl`

## Notes

- If real-time mode says it can’t find a teacher model, train an Isolation Forest once via the CLI first.
- Stopping: pressing Ctrl+C in the CLI will stop running real-time detection and return you to the menu.
