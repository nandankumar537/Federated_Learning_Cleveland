# Federated Learning with Flower and the Cleveland Dataset

This project demonstrates a Federated Learning (FL) setup using the Flower framework with the Cleveland Heart Disease dataset.  
The goal is to simulate multiple clients training collaboratively on subsets of the dataset without sharing raw data,  
preserving privacy while building a global model.

---

## Features
- Federated Learning setup using Flower (flwr) framework  
- Distributed training on the Cleveland Heart Disease dataset  
- Support for local client-server simulation  
- Configurable number of clients and training rounds  
- Training metrics such as accuracy, loss, precision, recall, and F1-score are logged  

## Clone the Repository
```
git clone https://github.com/nandankumar537/Federated_Learning_Cleveland.git
```

## Create a virtual environment 
```
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```
## Requirements
Install dependencies with:  
```bash
pip install -r requirements.txt
```
## Running the Setup

1. Start the Flower server
```bash
python server.py
```
2. Start clients (in separate terminals)
```bash
python client.py --client_id=0 --num_clients=2 --data=cleveland.csv
python client.py --client_id=1 --num_clients=2 --data=cleveland.csv
```
Arguments:

```--client_id```: Unique ID for the client (```0``` to ```num_clients-1```)

```--num_clients```: Total number of clients in federation (default: 2)

```--data```: Path to Cleveland dataset CSV (default: ```cleveland.csv```)

```--server```: Address of the Flower server (default: 127.0.0.1:8080)

```--log_dir```: Directory where client logs will be saved (default: ```.```)
3. Watch server logs for global metrics in ```server_logs.txt``` and check ```client*_log.txt``` for client metrics.
