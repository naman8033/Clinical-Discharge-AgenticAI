import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from agents.extractor_agent import run_extraction_pipeline

staging = {}

def get_patient_id(filename):
    """extract patient id from filename"""
    return filename.split("_")[0].strip().upper()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def is_patient_ready(patient_id):
    """if all files are present or not"""
    patient = staging.get(patient_id)
    if not patient:
        return False
    return (patient["bills"] is not None
            and patient['doctor_reports'] is not None
            and patient['lab_reports'] is not None)

def trigger_pipeline(patient_id, files=None):
    """code for invoking langgraph"""
    payload = files or staging[patient_id]
    run_extraction_pipeline(patient_id, payload)

class MonitorHandle(FileSystemEventHandler):
    def on_moved(self, event):
        self.process_file(event.src_path)
    
    def on_created(self, event):
        self.process_file(event.src_path)

    def process_file(self, filepath):
        if os.path.isdir(filepath):
            return
        print(f"process_file is called: {filepath}")
        folder = os.path.basename(os.path.dirname(filepath))
        filename = os.path.basename(filepath)
        patient_id = get_patient_id(filename)
        print(f"Parsed patient_id={patient_id}, folder={folder}, file={filename}")
        
        if patient_id not in staging:
            staging[patient_id] = {
                "bills": None,
                "doctor_reports": None,
                "lab_reports": None
            }
      
        for file in ["bills", "doctor_reports", "lab_reports"]:
            if file in folder.lower():
                staging[patient_id][file] = filepath
                print(f"Staged {file} path for patient {patient_id}")
        
        print(f"Checking if all files are ready for patient {patient_id}: {staging[patient_id]}")
        if is_patient_ready(patient_id):
            print(f"Patient {patient_id} is ready! Triggering clinical pipeline...")
            patient_files = dict(staging[patient_id])
            staging[patient_id] = {
                "bills": None,
                "doctor_reports": None,
                "lab_reports": None
            }
            trigger_pipeline(patient_id, patient_files)
        else:
            missing = [f for f in ["bills", "doctor_reports", "lab_reports"] if staging[patient_id][f] is None]
            print(f"Patient {patient_id} is not ready yet. Missing files: {missing}")


def start_monitor():
    paths = [
        os.path.join(_PROJECT_ROOT, "Data", "incoming", "bills"),
        os.path.join(_PROJECT_ROOT, "Data", "incoming", "doctor_reports"),
        os.path.join(_PROJECT_ROOT, "Data", "incoming", "lab_reports")
    ]

    observer = Observer()
    handler = MonitorHandle()

    for path in paths:
        print("paths", path)
        if not os.path.exists(path):
            print("path not exist")
            os.makedirs(path, exist_ok=True)
        
        observer.schedule(handler, path, recursive=False)
        
    observer.start()
    print("Monitor agent started")

    # Initial scan of already-uploaded files
    print("Scanning incoming folders for existing patient files...")
    for path in paths:
        if os.path.exists(path):
            for fname in os.listdir(path):
                fpath = os.path.join(path, fname)
                if os.path.isfile(fpath):
                    handler.process_file(fpath)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()