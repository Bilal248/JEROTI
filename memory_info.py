import psutil
import time

# Get details about physical memory usage
memory_info = psutil.virtual_memory()

print(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
print(f"Used Memory: {memory_info.used / (1024**3):.2f} GB")
print(f"Percentage Used: {memory_info.percent}%")

# Or monitor continuously in a loop
try:
    while True:
        print(f"Current Memory Usage: {psutil.virtual_memory().percent}%", end='\r')
        time.sleep(5)
except KeyboardInterrupt:
    pass