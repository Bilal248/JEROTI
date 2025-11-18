import os
import signal
import logging

logging.basicConfig(filename="anomaly.log", level=logging.INFO)

def handle_anomaly(pinfo, score):
    name = pinfo["name"]
    pid  = pinfo["pid"]

    logging.info(f"Anomaly detected — PID {pid}, {name}, score={score}")

    # Example actions
    if score < -0.3:
        try:
            os.kill(pid, signal.SIGKILL)
            logging.info(f"Killed suspicious process: {pid} ({name})")
        except:
            logging.info(f"Failed to kill: {pid}")