import sys

class WorkerLogger:
    def __init__(self, worker_port):
        self.terminal = sys.stdout
        self.log = open(f"worker_{worker_port}.txt", "w", encoding="utf-8")  # Overwrite each run

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure real-time updates

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

# Get worker port from command-line arguments
worker_port = sys.argv[1] if len(sys.argv) > 1 else "unknown"

# Redirect stdout and stderr to worker-specific log file
sys.stdout = WorkerLogger(worker_port)
sys.stderr = sys.stdout

import socket
import sys
import pickle
import torch
import torch.nn as nn
import os
import time
from values import MASTER_SERVER_IP, MASTER_REGISTER_PORT, MASTER_RESULT_PORT

os.system('clear')

MASTER_IP = MASTER_SERVER_IP
MASTER_PORT = MASTER_REGISTER_PORT
RESULT_PORT = MASTER_RESULT_PORT

worker_port = int(sys.argv[1])

print(f"Worker {worker_port} starting...")

# Function to send an error packet to master
def send_error_to_master(error_message):
    """Sends an error packet to the master node."""
    error_data = {"worker": worker_port, "error": error_message}
    serialized_error = pickle.dumps(error_data)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as result_socket:
        try:
            result_socket.connect((MASTER_IP, RESULT_PORT))
            result_socket.sendall(len(serialized_error).to_bytes(4, "big"))
            result_socket.sendall(serialized_error)
            print(f"Worker {worker_port} sent error report to master.")
        except Exception as e:
            print(f"Worker {worker_port} failed to send error to master: {e}")

def register_worker_with_master(worker_port):
    """Continuously tries to register the worker with the master node until successful."""
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(str(worker_port).encode(), (MASTER_IP, MASTER_PORT))
                print(f"Worker {worker_port} sent registration request.")
                return  # Exit loop upon success
        except Exception as e:
            print(f"Worker {worker_port} failed to register: {e}")
            send_error_to_master(f"Registration failed: {e}")
        
        print(f"Retrying registration in 5 seconds...")
        time.sleep(5)

# Call the function
register_worker_with_master(worker_port)
# Listen for model parts
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", worker_port))
        s.listen(1)
        print(f"Worker {worker_port} waiting for model part...")

        conn, addr = s.accept()
        print(f"Connection established with master!")

        # **Step 1: Receive the full data length (first 4 bytes)**
        data_length = int.from_bytes(conn.recv(4), "big")  # Read first 4 bytes
        data = b""

        # **Step 2: Receive the model data in chunks**
        while len(data) < data_length:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        # **Step 3: Deserialize only if fully received**
        if len(data) == data_length:
            try:
                model_part = pickle.loads(data)
                print(f"Worker {worker_port} successfully received model part!")
            except pickle.UnpicklingError:
                error_msg = "Failed to unpickle model part. Data may be incomplete."
                print(error_msg)
                send_error_to_master(error_msg)
                sys.exit(1)
        else:
            error_msg = f"Model part received is incomplete. Expected {data_length} bytes, got {len(data)} bytes."
            print(error_msg)
            send_error_to_master(error_msg)
            sys.exit(1)

        # Find the first input layer
        first_layer = next((layer for layer in model_part if isinstance(layer, (nn.Linear, nn.Conv2d))), None)

        if first_layer is None:
            error_msg = "No valid input layer (Linear/Conv2d) found."
            print(error_msg)
            send_error_to_master(error_msg)
            sys.exit(1)

        # Generate dummy input
        if isinstance(first_layer, nn.Linear):
            x_sample = torch.randn(4, first_layer.in_features)  # Batch size = 4
        elif isinstance(first_layer, nn.Conv2d):
            _, in_channels, kernel_height, kernel_width = first_layer.weight.shape
            x_sample = torch.randn(4, in_channels, kernel_height * 2, kernel_width * 2)  # Assume 2x the kernel size
        else:
            error_msg = "First layer is not Linear/Conv2d. Skipping sample run."
            print(error_msg)
            send_error_to_master(error_msg)
            x_sample = None

        # Run model if possible
        if x_sample is not None:
            try:
                output = model_part(x_sample)
                print(f"Worker {worker_port} processed data: {output.shape}")
            except Exception as e:
                error_msg = f"Model execution failed: {e}"
                print(error_msg)
                send_error_to_master(error_msg)
                sys.exit(1)

        # **Retry sending results up to 5 times**
        retries = 0
        while retries < 5:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as result_socket:
                try:
                    print(1)
                    result_socket.connect((MASTER_IP, RESULT_PORT))
                    print(2)
                    # Serialize result with length prefix
                    serialized_result = pickle.dumps(output)
                    result_socket.sendall(len(serialized_result).to_bytes(4, "big"))
                    result_socket.sendall(serialized_result)

                    print(f"Worker {worker_port} successfully sent results to master.")
                    break  # Exit retry loop on success
                except Exception as e:
                    print(f"Worker {worker_port} failed to send results (Attempt {retries + 1}/5): {e}")
                    retries += 1
                    time.sleep(2)
        else:
            error_msg = "Worker failed to send results after 5 attempts."
            print(error_msg)
            send_error_to_master(error_msg)

except Exception as e:
    error_msg = f"Unexpected failure: {e}"
    print(error_msg)
    send_error_to_master(error_msg)
    sys.exit(1)

print(f"Worker {worker_port} finished processing.")
