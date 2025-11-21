import psutil
import time
import os
import sys
import subprocess

def convert_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def get_extra_battery_info():
    if os.name == "nt":
        # Windows
        try:
            output = subprocess.check_output(["wmic", "path", "Win32_Battery", "get", "/format:list"]).decode()
            return output
        except:
            return "Advanced battery info not available (Windows)."

    elif sys.platform == "darwin":
        # macOS
        try:
            pmset = subprocess.check_output(["pmset", "-g", "batt"]).decode()
            ioreg = subprocess.check_output(["ioreg", "-r", "-c", "AppleSmartBattery"]).decode()
            return pmset + "\n" + ioreg
        except:
            return "Advanced battery info not available (macOS)."

    else:
        # Linux
        base = "/sys/class/power_supply/BAT0"
        info = ""
        if os.path.exists(base):
            for file in os.listdir(base):
                try:
                    with open(os.path.join(base, file), "r") as f:
                        info += f"{file}: {f.read().strip()}\n"
                except:
                    continue
        return info or "Advanced battery info not available (Linux)."

# Main execution


battery = psutil.sensors_battery()

print("\n=== Battery Information After 5 Minutes ===")
if battery:
    print(f"Battery percentage: {battery.percent}%")
    print(f"Power plugged in: {battery.power_plugged}")
    if battery.secsleft == psutil.POWER_TIME_UNLIMITED:
        print("Battery left: Unlimited (charging)")
    else:
        print(f"Battery left: {convert_time(battery.secsleft)}")
    print("\n--- Extra battery information ---")
    print(get_extra_battery_info())
    print("Waiting 5 minutes before displaying battery info...")
    time.sleep(300)  # 5 minutes = 300 seconds
    
else:
    print("Battery information not available.")
