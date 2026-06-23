import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

staging = {}

def get_patient_id(filename):
    """extract patient id from filename"""
    return filename.split("_")[0]

def is_patient_ready(patient_id):
    """if all files are present or not"""
    patient = staging[patient_id]
    print("is pateint ready : ",staging[patient_id])
    return (patient["bills"] is not None
            and patient['doctor_reports'] is not None
            and patient['lab_reports'] is not None)

def trigger_pipeline(patient_id):
    """code for invoiking langgraph"""
    print("langgraph invokation")
    pass

class MonitorHandle(FileSystemEventHandler):
    def on_any_event(self,event):
        print(event.event_type,event.src_path,event.is_directory)
    
    def on_moved(self,event):
        self.process_file(event.src_path)
    
    def on_created(self,event):
        self.process_file(event.src_path)

    def process_file(self,filepath):
        print("processing file function")
        if os.path.isdir(filepath):
            return
        filename = os.path.basename(filepath)

        print(f"\n new file detected:{filename}")
        patient_id = get_patient_id(filename)
        if patient_id not in staging:
            staging[patient_id] = {
                "bills":None,
                "doctor_reports":None,
                "lab_reports":None
            }
        
        for file in ["bills","doctor_reports","lab_reports"]:
            if file in filename.lower():
                staging[patient_id][file] = filepath
            
        print(f"current state for {patient_id}: {staging[patient_id]}")

        if is_patient_ready(patient_id):
            trigger_pipeline(patient_id)


def start_monitor():
    paths = [
        "./Data/incoming/bills",
        "./Data/incoming/doctor_reports",
        "./Data/incoming/lab_reports"
    ]

    observer = Observer()
    handler = MonitorHandle()

    for path in paths:
        print("paths",path)
        if not os.path.exists(path):
            print("path not exist")
            os.makedirs(path)
        
        observer.schedule(handler,path,recursive=False)
        
    observer.start()
    print("Monitor agent started")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()