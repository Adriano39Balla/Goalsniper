# Runs only the scheduler in a separate process.
import os, time
os.environ.setdefault("RUN_SCHEDULER", "1")
import main  # importing starts scheduler (your main.py does _start_scheduler_once())
print("Scheduler worker started.")
while True:
    time.sleep(3600)
