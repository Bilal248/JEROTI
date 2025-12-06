import subprocess
import time

PROCESS_NAME = "Safari"
INTERVAL = 1  # seconds

while True:
    # Get CPU percentage of the process
    result = subprocess.run(
        ["ps", "-A", "-o", "%cpu,pid,comm"],
        capture_output=True, text=True
    )
    
    cpu_usage = 0.0
    for line in result.stdout.splitlines():
        if PROCESS_NAME in line:
            cpu_usage += float(line.split()[0])
    
    # Approximate CPU power (MacBook Pro ~15W per CPU package)
    CPU_POWER_WATTS = 15
    process_power = CPU_POWER_WATTS * (cpu_usage / 100)
    
    print(f"{PROCESS_NAME}: CPU={cpu_usage:.2f}% ~ Power={process_power:.2f} W")
    time.sleep(INTERVAL)
