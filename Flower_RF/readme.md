# Federated Random Forest with Flower (Cleveland Dataset)

This project implements a **Federated Learning client** using a **Random Forest Classifier** on the Cleveland Heart Disease dataset.  
The client connects to a central Flower server, trains locally on its partition of the dataset, and reports evaluation metrics,  
while raw data never leaves the client device.

---

## Features
- Random Forest (`scikit-learn`) integrated with Flower client API  
- Preprocessing: missing value imputation + standardization  
- Automatic dataset partitioning across clients using Stratified K-Folds  
- Local train/validation split per client  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, Mean CV Accuracy  
- Logs written per client for each training round  

## Clone the Repository
```
git clone https://github.com/nandankumar537/Federated_Learning_Cleveland.git
cd Flower_RF
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

Random Forest parameters (```max_depth```, ```n_estimators```, etc.) can be tuned inside SklearnRFClient.

To increase federation size, adjust ```--num_clients``` and launch additional clients.

By default, parameter aggregation is disabled since Random Forests are not naturally aggregatable in Flower.

For data splits, modify the ```converter.py``` file
