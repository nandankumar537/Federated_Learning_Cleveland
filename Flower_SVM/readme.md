# Federated SVM with Flower (Cleveland Dataset)

This project implements a **Flower federated client** using an **SVM classifier** (`scikit-learn`) trained locally on partitions of the Cleveland Heart Disease dataset.  
The system supports **client-side Differential Privacy (DP)** using update clipping + Gaussian noise applied to an aggregated vector created from **predict_proba outputs on shared public data**.

Because SVMs with RBF kernels do not expose learnable model weights suitable for federated averaging, the project performs **prediction-aggregation FL**, identical to the Random Forest case.

---

## Features
- SVM client using Flower’s `NumPyClient` interface.
- Preprocessing: missing value imputation + standardization.
- Stratified client-wise data partitioning.
- Local train/validation split inside each client.
- Metrics logged each round: Accuracy, Precision, Recall, F1, Mean CV Accuracy.
- Per-client logs stored in `client{ID}_log.txt`.
- Optional **client-side Differential Privacy** added to the prediction vector update.
- Supports a shared public dataset (`X_public.npy`) used for consistent prediction vector generation across clients.

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

3. Starting Server (DP)
```python
python server.py --public_data X_public.npy --num_rounds 5 --n_classes 2
```
Without public vector initialization
```python
python server.py --num_rounds 5
```

4. Starting Clients (DP)
```python
python client_svm_dp.py \
    --client_id 0 \
    --num_clients 3 \
    --data cleveland.csv \
    --public_data X_public.npy \
    --dp_enabled \
    --dp_epsilon 1.0 \
    --dp_delta 1e-5 \
    --clip_norm 1.0

```
Arguments:

```--client_id```: Unique ID for the client (```0``` to ```num_clients-1```)

```--num_clients```: Total number of clients in federation (default: 2)

```--data```: Path to Cleveland dataset CSV (default: ```cleveland.csv```)

```--server```: Address of the Flower server (default: 127.0.0.1:8080)

```--log_dir```: Directory where client logs will be saved (default: ```.```)

```--public_data``` :Path to shared public data (.npy).
                Required if prediction-aggregation FL is used.
                The same file must be provided to all clients and server.

```-dp_enabled```:Enables client-side DP (clip + Gaussian noise).

```--dp_epsilon```:Per-round epsilon for DP Gaussian mechanism (default=```1.0```).

```--dp_delta``` :Per-round delta for DP Gaussian mechanism (default=```1e-5```).

```--clip_norm```:L2 clip norm used to bound Δ during DP update (default=```1.0```).

3. Watch server logs for global metrics in ```server_logs.txt``` and check ```client*_log.txt``` for client metrics.

## Notes

SVMs do not expose coefficients for kernel methods, so model parameters cannot be aggregated in the usual FedAvg sense.

This setup demonstrates federated orchestration + metric logging rather than true parameter sharing.

For federated aggregation, models with explicit weights (e.g., Logistic Regression, Neural Networks) are better suited.

You can increase the number of clients by adjusting ```--num_clients``` and running additional client processes.

For data splits, modify the ```converter.py``` file


