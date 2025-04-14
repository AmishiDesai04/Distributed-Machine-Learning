import subprocess
import time
from master import detect_workers
from model import split_model, load_user_model

# Number of required workers
required_workers = 3  # Change this if needed

# Detect currently running workers
existing_worker_ports = detect_workers()

# Start missing workers
new_worker_ports = [5001 + i for i in range(required_workers) if (5001 + i) not in existing_worker_ports]

for port in new_worker_ports:
    print(f"🚀 Starting worker on port {port}...")
    subprocess.Popen(["python", "worker.py", str(port)])
    time.sleep(2)  # Give some time for workers to start

# Wait a few seconds to let workers initialize
time.sleep(5)

# Start master node
print("🚀 Starting master node...")
subprocess.Popen(["python", "master.py"])
