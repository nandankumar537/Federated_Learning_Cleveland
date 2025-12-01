# Federated Logistic Regression with Flower (Cleveland Dataset)

This project implements a **Flower client** that trains a **Logistic Regression** model (`scikit-learn`) on partitions of the Cleveland Heart Disease dataset and participates in federated learning rounds coordinated by a Flower server.  
The system supports **optional client-side Differential Privacy (DP)** using update clipping and Gaussian noise.

---

## Features
- Logistic Regression federated across clients using Flower.
- Missing-value imputation + feature standardization.
- Stratified dataset partitioning across clients.
- Local train/validation split inside each client.
- Metrics per round: Accuracy, Precision, Recall, F1, Mean CV Accuracy.
- Per-client logs saved as `client{ID}_log.txt`.
- Optional **client-side Differential Privacy** (clip-and-noise on model updates).

---

## Clone the Repository
```
git clone https://github.com/nandankumar537/Federated_Learning_Cleveland.git
cd Flower_LR
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

3. Start clients with DP enabled (in separate terminals)
```bash
python client.py --client_id 0 --num_clients 3 --dp_enabled --dp_epsilon 1.0 --dp_delta 1e-5 --clip_norm 1.0
python client.py --client_id 1 --num_clients 3 --dp_enabled --dp_epsilon 1.0 --dp_delta 1e-5 --clip_norm 1.0
python client.py --client_id 2 --num_clients 3 --dp_enabled --dp_epsilon 1.0 --dp_delta 1e-5 --clip_norm 1.0
```

Arguments:

```--client_id```: Unique ID for the client (```0``` to ```num_clients-1```)

```--num_clients```: Total number of clients in federation (default: ```2```)

```--data```: Path to Cleveland dataset CSV (default: ```cleveland.csv```)

```--server```: Address of the Flower server (default: ```127.0.0.1:8080```)

```--log_dir```: Directory where client logs will be saved (default: ```.```)

```--dp_enabled```:Enables client-side Differential Privacy (clipping + Gaussian noise).

```--dp_epsilon```:Per-round DP epsilon for Gaussian mechanism (default=```1.0```).

```--dp_delta```:Per-round DP delta for Gaussian mechanism (default=```1e-5```)

```--clip_norm```:L2 clip norm applied to model update before adding noise (default=```1.0```).

4. Watch server logs for global metrics in ```server_logs.txt``` and check ```client*_log.txt``` for client metrics.

## Notes

You can increase the number of clients by adjusting ```--num_clients``` and running additional client processes.

For data splits, modify the ```converter.py``` file
