import psutil
import time
import logging


skip_Processes =  [
  "universalaccessd",
  "CategoriesService",
  "com.apple.Safari.SafeBrowsing.Service",
  "PlugInLibraryService",
  "SetStoreUpdateService",
  "ServiceExtension",
  "RemoteManagementAgent",
  "SpeechSynthesisServerXPC",
  "remindd",
  "AccessibilityUIServer",
  "FinderSyncExtension",
  "ControlStrip",
  "com.apple.dock.extra",
  "AirPlayUIAgent",
  "JournalWidgetsSecure",
  "FindMyWidgetPeople",
  "VoiceMemosSettingsWidgetExtension",
  "MobileTimerIntents",
  "ScreenTimeWidgetExtension",
  "WeatherIntents",
  "RecordWidgetExtension",
  "FindMyWidgetItems",
  "ScreenTimeWidgetIntentsExtension",
  "HomeEnergyWidgetsExtension",
  "com.apple.Notes.WidgetExtension",
  "PeopleWidget_macOSExtension",
  "RemindersWidgetExtension",
  "Google Drive Helper",
  "DFSFileProviderExtension",
  "universalAccessAuthWarn",
  "Google Drive Helper (Renderer)",
  "sirittsd",
  "coreautha",
  "itunescloudd",
  "cfprefsd",
  "ctkd",
  "inputanalyticsd",
  "IFTranscriptSELFIngestor",
  "OfficeThumbnailExtension",
  "ShortcutLiveActivityWidgetExtension",
  "VTDecoderXPCService",
  "zsh",
  "Code Helper (Plugin)",
  "CGPDFService",
  "PowerChime",
  "intelligenceplatformd",
  "proactiveeventtrackerd",
  "TextThumbnailExtension",
  "Google Chrome Helper (Renderer)",
  "UserNotificationCenter",
  "LinkedNotesUIService",
  "ControlCenterHelper",
  "AudiovisualThumbnailExtension",
  "MEGAShellExtFinder",
  "VTEncoderXPCService",
  "com.apple.appkit.xpc.openAndSavePanelService",
  "WebThumbnailExtension",
  "ReportCrash",
  "deleted",
  "networkserviceproxy",
  "managedappdistributionagent",
  "sharedfilelistd",
  "AMPDeviceDiscoveryAgent",
  "bird",
  "diagnostics_agent",
  "appstoreagent",
  "siriactionsd",
  "com.apple.quicklook.ThumbnailsAgent",
  "amsaccountsd",
  "CommCenter",
  "pkd",
  "com.apple.hiservices-xpcservice",
  "extensionkitservice",
  "OfficeThumbnailExtension",
  "WebThumbnailExtension",
  "com.apple.WebKit.GPU",
  "ReportCrash"
]

logging.basicConfig(filename="anomaly.log", level=logging.INFO)

# running stats: {pid: {"cpu": (mean, std, n), "rss": (...), "threads": (...) }}
running_stats = {}

THRESHOLD = 3  # number of standard deviations to flag anomaly

def update_stats(pid, feature, value):
    mean, std, n = running_stats[pid].get(feature, (0, 0, 0))
    n += 1
    new_mean = mean + (value - mean) / n
    new_std = ((n - 1)*(std**2) + (value - mean)*(value - new_mean)) / n
    new_std = new_std**0.5
    running_stats[pid][feature] = (new_mean, new_std, n)
    return new_mean, new_std

def handle_anomaly(pinfo, feature, value, score):
    name = pinfo["name"]
    pid = pinfo["pid"]
    logging.info(f"Anomaly detected: {name} (PID {pid}) feature={feature}, value={value}, score={score}")
    # Example action: kill process if CPU spike
    if feature == "cpu" and value > 50 and name and name.lower() not in (s.lower() for s in skip_Processes):
        try:
            import os, signal
            # os.kill(pid, signal.SIGKILL)
            logging.info(f"Killed process {name} (PID {pid})")
        except Exception:
            logging.info(f"Failed to kill {name} (PID {pid})")

def run():
    print("Simple online anomaly detector started...")
    while True:
        for p in psutil.process_iter(['pid','name','cpu_percent','memory_info','num_threads']):
            try:
                pid = p.info['pid']
                if pid not in running_stats:
                    running_stats[pid] = {}
                features = {
                    "cpu": p.info['cpu_percent'],
                    "rss": p.info['memory_info'].rss,
                    "threads": p.info['num_threads']
                }
                for f, v in features.items():
                    mean, std = update_stats(pid, f, v)[:2]
                    if std > 0 and abs(v - mean)/std > THRESHOLD:
                        handle_anomaly(p.info, f, v, (v - mean)/std)
            except:
                continue
        time.sleep(2)

if __name__ == "__main__":
    run()



