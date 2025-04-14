import logging
import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
from dask.distributed import Client, get_worker
import requests
import time
import threading

class Logger:
    def __init__(self, filename="master_a.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message + "\n")  # Ensures a newline after each log entry
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger("master_a.txt")
sys.stderr = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s\n",
    handlers=[
        logging.FileHandler("master_a.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

flask_logger = logging.getLogger("werkzeug")
flask_logger.addHandler(logging.FileHandler("master_a.txt", mode="w", encoding="utf-8"))
flask_logger.setLevel(logging.INFO)

os.environ["FLASK_ENV"] = "production"

logger = logging.getLogger(__name__)
app = Flask(__name__)
global_model = None
df = pd.read_csv("enhanced_anxiety_dataset.csv")

features = ["Age", "Sleep Hours", "Physical Activity (hrs/week)", "Caffeine Intake (mg/day)",
            "Stress Level (1-10)", "Heart Rate (bpm)", "Breathing Rate (breaths/min)",
            "Sweating Level (1-5)", "Diet Quality (1-10)"]

target = "Anxiety Level (1-10)"

X = df[features].values
y = df[target].values

def train_model_sequentially(X_chunks, y_chunks):
    model = LinearRegression()
    worker = get_worker()
    log_messages = []

    for i in range(len(X_chunks)):
        msg = f"\n\nWorker {worker.address}: -- Training on chunk {i+1} with {len(X_chunks[i])} samples\n"
        logger.info(msg)
        log_messages.append(msg)

        model.fit(X_chunks[i], y_chunks[i])

        msg = f"\n\nWorker {worker.address}: Completed training chunk {i+1}\n"
        logger.info(msg)
        log_messages.append(msg)

    msg = f"\n\n ===================================== \n Worker {worker.address}: \n\nFinal model coefficients \t {model.coef_}, \nIntercept {model.intercept_}\n ===================================== \n"
    logger.info(msg)
    log_messages.append(msg)

    return model.coef_, model.intercept_, log_messages

@app.route('/train', methods=['POST'])
def train():
    global global_model
    num_devices = len(client.scheduler_info()['workers'])
    X_chunks = np.array_split(X, num_devices)
    y_chunks = np.array_split(y, num_devices)

    logger.info(f"\nMaster Node: Distributing training to {num_devices} workers\n")

    futures = []
    for i in range(num_devices):
        future = client.submit(train_model_sequentially, [X_chunks[i]], [y_chunks[i]])
        futures.append(future)
        logger.info(f"\n===================================== \nMaster Node: Sent chunk {i+1} to worker\n")

    results = client.gather(futures)

    coefficients = np.mean([res[0] for res in results], axis=0)
    intercept = np.mean([res[1] for res in results])
    worker_logs = [log for res in results for log in res[2]]

    global_model = LinearRegression()
    global_model.coef_ = coefficients
    global_model.intercept_ = intercept

    logger.info(f"\n\n ===================================== \nMaster Node: Model training completed.\n ===================================== \nCoefficients: \t {coefficients},\n Intercept: {intercept}\n")
    for log in worker_logs:
        logger.info(log + "\n")

    return jsonify({
        "message": "Model trained successfully",
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "worker_logs": worker_logs
    })

@app.route('/predict', methods=['POST'])
def predict():
    global global_model
    if global_model is None:
        return jsonify({"error": "Model is not trained yet"}), 400

    data = request.get_json()
    X_new = np.array(data["X"]).reshape(-1, len(features))
    worker = get_worker()
    predictions = global_model.predict(X_new)

    logger.info(f"\nWorker {worker.address}: Made predictions: {predictions}\n")
    return jsonify({"predictions": predictions.tolist()})

def run_flask_app():
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    client = Client("tcp://10.125.46.236:8786")
    logger.info(f"\nMaster Node: Connected to Dask cluster with workers: {client.scheduler_info()['workers']}\n")

    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    logger.info("\nWaiting for Flask app to fully start...\n")
    time.sleep(5)

    train_url = "http://10.125.46.236:5000/train"
    logger.info(f"\nMaster Node: Triggering training via POST request to {train_url}\n")

    try:
        response = requests.post(train_url)
        if response.status_code == 200:
            logger.info("\nMaster Node: Training triggered successfully.\n")
        else:
            logger.error(f"\nMaster Node: Failed to trigger training. Status Code: {response.status_code}\n")
    except Exception as e:
        logger.error(f"\nMaster Node: Error triggering training: {e}\n")

    logger.info("\nMain process continues, Flask app running in background.\n")
