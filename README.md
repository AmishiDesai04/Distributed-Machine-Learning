# Distributed-Machine-Learning Proof Of Concept 

This project demonstrates a distributed machine learning system that leverages model and data parallelism across multiple machines. It includes two primary implementations: a CNN-based inference system using model parallelism and a Linear Regression system using data parallelism with Dask.

## Overview

The architecture is designed to run over a local area network. A master node coordinates tasks, distributes models or data, and aggregates results from multiple worker nodes. Communication is handled using UDP for worker registration and TCP for data exchange.

## Key Features

- Distributed CNN model using model parallelism
- Linear Regression training using Dask and data parallelism
- REST API built with Flask for initiating training and making predictions
- Lightweight and runs on low-spec machines over LAN

## Technologies Used

- Python 3.10
- PyTorch
- Dask
- Flask
- Scikit-learn
- NumPy, Pandas
- TCP/UDP socket communication
- Pickle for serialization


## Project Structure

- `master.py` – Controls model/data distribution, API server
- `worker.py` – Handles assigned training or inference tasks
- `flask_server.py` – Hosts Flask routes for `/train` and `/predict`
- `utils/` – Utility functions for serialization and configuration
- `model.pkl` – Sample CNN model (PyTorch)
- `dataset.csv` – Input data for regression
- `requirements.txt` – List of dependencies
- `bankend_81.py` - Linear Regression Distributed Model

## Usage

- Use `/train` API endpoint to start distributed training.
- Use `/predict` API endpoint to send input data and get predictions.

## Contributors

[@AmishiDesai04](https://github.com/AmishiDesai04), [@chahelgupta](https://github.com/chahelgupta), [@vpratham](https://github.com/vpratham)

