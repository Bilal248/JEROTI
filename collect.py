import csv
import psutil
import time
import os
import datetime


def collect_and_save(duration=15, interval=1):

    proceses = 0

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    DATA_PATH = f"data/{duration}seconds-{date}.csv"

    header = ["timestamp", "pid", "name", "cpu", "rss", "threads"]

    write_header = not os.path.exists(DATA_PATH)

    end_time = time.time() + duration

    bar_length = 50

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        sec = 0
        while time.time() < end_time:
            sec += 1
            progress = sec / duration
            percent = int(progress * 100)

            filled = int(bar_length * progress)
            bar = "[" + "=" * filled + "-" * \
                (bar_length - filled) + f"] {percent:3d}%"

            print(bar, end="\r", flush=True)

            for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "num_threads"]):
                try:
                    proceses = proceses + 1
                    cpu = p.info.get("cpu_percent", 0.0)
                    mem = p.info.get("memory_info").rss if p.info.get(
                        "memory_info") else 0
                    threads = p.info.get("num_threads", 0)

                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        p.info.get("pid"),
                        p.info.get("name"),
                        cpu,
                        mem,
                        threads
                    ])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            time.sleep(interval)

    print()
    print(f"\n\tData Collected Total Rows {proceses} \n\tSaved to {DATA_PATH}")
