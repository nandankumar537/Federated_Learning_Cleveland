# Federated Random Forest with Flower (Cleveland Dataset)

This project implements a **Flower client** that trains a **Random Forest** model (`scikit-learn`) on partitions of the Cleveland Heart Disease dataset and participates in federated learning rounds coordinated by a Flower server.  
Because Random Forests are non-parametric, the project implements **client-update DP on model predictions** (prediction-aggregation on a shared public calibration set) as the privacy-preserving mechanism.

---

## Features
- Random Forest (`scikit-learn`) clients participating in federated rounds using Flower.
- Missing-value imputation + standardization preprocessing pipeline.
- Stratified dataset partitioning across clients.
- Local train / validation split per client.
- Metrics per round: Accuracy, Precision, Recall, F1, Mean CV Accuracy.
- Per-client logs saved as `client{ID}_log.txt`.
- Optional client-side DP (clip + Gaussian noise) applied to flattened `predict_proba(X_public)` vectors (prediction-aggregation).
- Server can initialize a global prediction vector from a shared public dataset (`X_public.npy`).

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

3. Running the server (DP)

If you plan to run prediction-aggregation mode, generate ```X_public.npy``` first Using ```python prepare_data.py```
 Then run:
```python
    python server.py --public_data X_public.npy --num_rounds 5 --n_classes 2
```


If you do not want to use public-data aggregation (clients operate locally only and send no parameters), omit ```--public_data```
Run : 
```python
python server.py --num_rounds 5
```

Arguments:

```--client_id```: Unique ID for the client (```0``` to ```num_clients-1```)

```--num_clients```: Total number of clients in federation (default: 2)

```--data```: Path to Cleveland dataset CSV (default: ```cleveland.csv```)

```--server```: Address of the Flower server (default: 127.0.0.1:8080)

```--log_dir```: Directory where client logs will be saved (default: ```.```)


```--public_data```:Path to shared public data (NumPy `.npy`) used to compute predict_proba.
                If omitted, client will not participate in prediction-aggregation.

```--dp_enabled```: Enables client-side Differential Privacy (clipping + Gaussian noise) on the prediction-vector updates.

```--dp_epsilon```: Per-round DP epsilon for Gaussian mechanism (default=```1.0```).

```--dp_delta```: Per-round DP delta for Gaussian mechanism (default=```1e-5```).

```--clip_norm```: L2 clip norm applied to the prediction-vector update before adding noise (default=```1.0```).


4. Running clients (DP)

3 clients, prediction-aggregation with DP enabled
```python
python client.py --client_id 0 --num_clients 3 --data cleveland.csv --public_data X_public.npy --dp_enabled --dp_epsilon 1.0 --dp_delta 1e-5 --clip_norm 1.0 --server 127.0.0.1:8080
python client.py --client_id 1 --num_clients 3 --data cleveland.csv --public_data X_public.npy --dp_enabled --dp_epsilon 1.0 --dp_delta 1e-5 --clip_norm 1.0 --server 127.0.0.1:8080
python client.py --client_id 2 --num_clients 3 --data cleveland.csv --public_data X_public.npy --dp_enabled --dp_epsilon 1.0 --dp_delta 1e-5 --clip_norm 1.0 --server 127.0.0.1:8080

```




3. Watch server logs for global metrics in ```server_logs.txt``` and check ```client*_log.txt``` for client metrics.

## Notes

Random Forest parameters (```max_depth```, ```n_estimators```, etc.) can be tuned inside SklearnRFClient.

To increase federation size, adjust ```--num_clients``` and launch additional clients.

By default, parameter aggregation is disabled since Random Forests are not naturally aggregatable in Flower.

For data splits, modify the ```converter.py``` file
