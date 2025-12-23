import csv
import psutil
import time
import os
import datetime
import json

def collect_and_save(duration=15, interval=1):
    """CLI collector: save CSV and JSON simultaneously"""
    proceses = 0
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = f"data/{duration}seconds-{date}.csv"
    json_path = f"/Users/apple/Desktop/J.E.R.O.T.I..O/JEROTI_App/JEROTI_App/ProcessesData/{duration}seconds-{date}.json"
    os.makedirs("data", exist_ok=True)

    header = ["timestamp", "pid", "name", "cpu", "rss", "threads"]
    write_header_csv = not os.path.exists(csv_path)
    write_header_json = not os.path.exists(json_path)

    end_time = time.time() + duration
    bar_length = 50

    json_data = []

    with open(csv_path, "a", newline="") as f_csv:
        writer = csv.writer(f_csv)
        if write_header_csv:
            writer.writerow(header)

        sec = 0
        while time.time() < end_time:
            sec += 1
            progress = sec / duration
            percent = int(progress * 100)
            filled = int(bar_length * progress)
            bar = "[" + "=" * filled + "-" * (bar_length - filled) + f"] {percent:3d}%"
            print(bar, end="\r", flush=True)

            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "num_threads"]):
                try:
                    proceses += 1
                    cpu = p.info.get("cpu_percent", 0.0)
                    mem = p.info.get("memory_info").rss if p.info.get("memory_info") else 0
                    threads = p.info.get("num_threads", 0)

                    row = [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        p.info.get("pid"),
                        p.info.get("name"),
                        cpu,
                        mem,
                        threads
                    ]
                    writer.writerow(row)

                    # Also append to JSON data
                    json_data.append({
                        "timestamp": row[0],
                        "pid": row[1],
                        "name": row[2],
                        "cpu": row[3],
                        "rss": row[4],
                        "threads": row[5]
                    })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            time.sleep(interval)

    # Save JSON file
    with open(json_path, "w") as f_json:
        json.dump({"processes": json_data}, f_json, indent=2)

    print()
    print(f"\n\tData Collected Total Rows {proceses} \n\tSaved to {csv_path} and {json_path}")
    return csv_path, json_path
