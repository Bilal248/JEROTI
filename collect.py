import psutil, time, csv

with open("../data/sample_training_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "pid", "name", "cpu", "rss", "threads"])

    while True:
        for p in psutil.process_iter(['pid','name','cpu_percent','memory_info','num_threads']):
            try:
                info = p.info
                writer.writerow([
                    time.time(),
                    info['pid'],
                    info['name'],
                    info['cpu_percent'],
                    info['memory_info'].rss,
                    info['num_threads']
                ])
            except:
                pass
        time.sleep(1)