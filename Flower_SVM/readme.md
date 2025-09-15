# Federated Learning with SVM and Flower (Cleveland Dataset)

This project demonstrates how to train a **Support Vector Machine (SVM)** in a **Federated Learning (FL)** setup using the [Flower](https://flower.dev/) framework and the Cleveland Heart Disease dataset.

Unlike linear models, SVMs with kernels do not have directly shareable parameters for aggregation, so this setup mainly showcases **local training with centralized orchestration** and reporting of evaluation metrics.

---

## Features
- **Federated setup** with Flower (client-server simulation)
- Local SVM training on each client
- Train/validation split per client
- Automatic partitioning of dataset among clients
- Metrics logged per round: Accuracy, Precision, Recall, F1-score, Mean CV Accuracy
- Each client saves logs in `client*_log.txt`  

## Clone the Repository
```
git clone https://github.com/nandankumar537/Federated_Learning_Cleveland.git
cd Flower_SVM
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

## Notes

SVMs do not expose coefficients for kernel methods, so model parameters cannot be aggregated in the usual FedAvg sense.

This setup demonstrates federated orchestration + metric logging rather than true parameter sharing.

For federated aggregation, models with explicit weights (e.g., Logistic Regression, Neural Networks) are better suited.

You can increase the number of clients by adjusting ```--num_clients``` and running additional client processes.

For data splits, modify the ```converter.py``` file


