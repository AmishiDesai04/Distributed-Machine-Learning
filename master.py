import sys

class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure log updates in real-time

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

sys.stdout = Logger("master.txt")


import socket
import pickle
import time
import threading
import torch
import math
from model import split_model, load_user_model
from tabulate import tabulate
import os
from values import MASTER_SERVER_IP, MASTER_REGISTER_PORT, MASTER_RESULT_PORT

os.system('cls' if os.name == 'nt' else 'clear')


MASTER_IP = MASTER_SERVER_IP
REGISTER_PORT = MASTER_REGISTER_PORT
RESULT_PORT = MASTER_RESULT_PORT

worker_ports = []
worker_ips = []
worker_status = {}
print("=====================================================================")
print("PHASE 1 - Worker Registration")
print("=====================================================================\n\n")
print("Waiting for workers to register...")

# Worker registration
def register_workers():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("0.0.0.0", REGISTER_PORT))
        s.settimeout(10)  # Timeout to avoid infinite waiting
        try:
            while True:
                data, addr = s.recvfrom(1024)
                worker_ip, _ = addr
                worker_port = int(data.decode())
                if worker_port not in worker_ports:
                    worker_ports.append(worker_port)
                    worker_ips.append(worker_ip)
                    worker_status[worker_port] = "Connected"
                    print(f"Worker {worker_port} registered from {worker_ip}!")
                    s.sendto(b"ACK", addr)
        except socket.timeout:
            if not worker_ports:
                print("No workers detected. Start workers first!")
                exit(1)

register_workers()

print("\n*********************************************************************")
print(f"Total {len(worker_ports)} workers detected: {worker_ports}")
print("*********************************************************************\n")

# Load model and calculate minimum workers required
model = load_user_model()
model_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Bytes
MIN_MEMORY_PER_WORKER = 50 * 1024 * 1024  # Assume 50MB per worker
min_workers = max(2, math.ceil(model_size / MIN_MEMORY_PER_WORKER))

print(f"Model estimated size: {model_size / (1024 * 1024):.2f} MB")
print(f"Minimum workers required: {min_workers}")

while len(worker_ports) < min_workers:
    print(f"Not enough workers! Required: {min_workers}, Available: {len(worker_ports)}")
    register_workers()

print("Waiting 10s before sending model to workers...")
time.sleep(10)

print()
# Display connected workers
headers = ["Worker #", "IP", "Port", "Status"]
data = [[i+1, worker_ips[i], worker_ports[i], worker_status[worker_ports[i]]] for i in range(len(worker_ports))]
print(tabulate(data, headers=headers, tablefmt="grid"))

print("\n\n=====================================================================")
print("PHASE 2 - Model Distribution")
print("=====================================================================\n")
model_parts = split_model(model, len(worker_ports))

# Send model parts
for i, worker_port in enumerate(worker_ports):
    print(f"Sending model part {i+1}/{len(worker_ports)} to Worker {worker_port}...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((worker_ips[i], worker_port))
            serialized_data = pickle.dumps(model_parts[i])

            # Send data length first
            s.sendall(len(serialized_data).to_bytes(4, 'big'))
            s.sendall(serialized_data)
            print(f"Model part {i+1} sent to Worker {worker_port}!")
        except Exception as e:
            worker_status[worker_port] = "Failed"
            print(f"Connection to Worker {worker_port} failed: {e}")

# Function to receive full data reliably
def receive_full_data(conn):
    try:
        data_length = int.from_bytes(conn.recv(4), 'big')  # Read first 4 bytes
        data = b""
        while len(data) < data_length:
            packet = conn.recv(4096)  # Receive in chunks
            if not packet:
                break
            data += packet
        return pickle.loads(data)
    except:
        return {"error": "Data corrupted"}

# Result collection
def listen_for_results():
    print("\n\n=====================================================================")
    print("PHASE 3 - Collecting Results")
    print("=====================================================================\n")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as result_socket:
        result_socket.bind(("0.0.0.0", RESULT_PORT))
        result_socket.listen(len(worker_ports))
        
        received_results = 0
        while received_results < len(worker_ports):
            conn, addr = result_socket.accept()
            result_data = receive_full_data(conn)

            if isinstance(result_data, dict) and "error" in result_data:
                print(f"Error from Worker {result_data['worker']}: {result_data['error']}")
                worker_status[result_data['worker']] = "Error"
            else:
                print(f"Received result from Worker {addr[1]}: {result_data}")
                worker_status[addr[1]] = "Completed"
                received_results += 1

# Start result collection in a separate thread
result_thread = threading.Thread(target=listen_for_results, daemon=True)
result_thread.start()

# Keep master alive while listening for results
while result_thread.is_alive():
    time.sleep(1)

# Display final worker statuses
print("\n=====================================================================")
print("FINAL STATUS")
print("=====================================================================\n")
data = [[i+1, worker_ips[i], worker_ports[i], worker_status[worker_ports[i]]] for i in range(len(worker_ports))]
print(tabulate(data, headers=headers, tablefmt="grid"))

print("\nTraining Complete.")
