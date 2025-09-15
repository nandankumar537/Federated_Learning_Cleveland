import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import flwr as fl

# ----- Configuration -----
FEATURE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET_COLUMN = "target"

def load_and_preprocess(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if isinstance(df.columns, pd.RangeIndex) and df.shape[1] == 14:
        df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[keep_cols]
    df = df.replace("?", np.nan)
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].fillna(0).astype(int) > 0).astype(int)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def partition_data(
    X: np.ndarray, y: np.ndarray, num_clients: int, client_id: int, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    for i, (_, idx_client) in enumerate(skf.split(X, y)):
        if i == client_id:
            return X[idx_client], y[idx_client]
    raise ValueError("Invalid client_id or partitioning failure")

def train_val_split(
    Xc: np.ndarray, yc: np.ndarray, val_ratio: float = 0.2, seed: int = 123
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    np.random.seed(seed)
    idx = np.arange(len(yc))
    np.random.shuffle(idx)
    val_size = int(len(yc) * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    return (Xc[train_idx], yc[train_idx]), (Xc[val_idx], yc[val_idx])

def mean_cv_accuracy_local(X_local: np.ndarray, y_local: np.ndarray, folds: int = 5) -> float:
    skf = StratifiedKFold(n_splits=min(folds, len(np.unique(y_local)) if len(y_local) >= folds else 2), shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf.split(X_local, y_local):
        Xtr, Xva = X_local[tr_idx], X_local[va_idx]
        ytr, yva = y_local[tr_idx], y_local[va_idx]
        clf = SVC(
            C=1,
            kernel='rbf',
            gamma='scale',
            degree=3,
            coef0=0,
            probability=True,
            random_state=42,
        )
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xva)
        accs.append(accuracy_score(yva, yhat))
    return float(np.mean(accs)) if accs else 0.0

class SklearnSVMClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xva: np.ndarray,
        yva: np.ndarray,
        X_local: np.ndarray,
        y_local: np.ndarray,
        log_dir: str = ".",
    ):
        self.client_id = client_id
        self.Xtr = Xtr
        self.ytr = ytr
        self.Xva = Xva
        self.yva = yva
        self.X_local = X_local
        self.y_local = y_local
        self.clf = SVC(
            C=1,
            kernel='rbf',
            gamma='scale',
            degree=3,
            coef0=0,
            probability=True,
            random_state=42,
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"client{client_id}_log.txt")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tMeanCVAccuracy\n")

    def get_parameters(self, config):
        # Not supported for scikit-learn SVM, just return empty list
        return []

    def set_parameters(self, parameters):
        # Not supported
        pass

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", -1))
        self.clf.fit(self.Xtr, self.ytr)
        return self.get_parameters(config={}), len(self.Xtr), {"server_round": server_round}

    def evaluate(self, parameters, config):
        server_round = int(config.get("server_round", -1))
        y_pred = self.clf.predict(self.Xva)
        acc = accuracy_score(self.yva, y_pred)
        prec = precision_score(self.yva, y_pred, zero_division=0)
        rec = recall_score(self.yva, y_pred, zero_division=0)
        f1 = f1_score(self.yva, y_pred, zero_division=0)
        mean_cv_acc = mean_cv_accuracy_local(self.X_local, self.y_local, folds=5)
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_cv_accuracy": float(mean_cv_acc),
            "server_round": server_round,
            "client_id": self.client_id,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{server_round}\t{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t"
                f"{metrics['recall']:.4f}\t{metrics['f1']:.4f}\t{metrics['mean_cv_accuracy']:.4f}\n"
            )
        return 0.0, len(self.Xva), metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cleveland.csv", help="Path to cleveland dataset CSV")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address host:port")
    parser.add_argument("--num_clients", type=int, default=2, help="Total number of clients in the federation")
    parser.add_argument("--client_id", type=int, required=True, help="Client id in [0..num_clients-1]")
    parser.add_argument("--log_dir", type=str, default=".", help="Directory to write client logs")
    args = parser.parse_args()

    X, y = load_and_preprocess(args.data)
    Xc, yc = partition_data(X, y, num_clients=args.num_clients, client_id=args.client_id, seed=42)
    (Xtr, ytr), (Xva, yva) = train_val_split(Xc, yc, val_ratio=0.2, seed=123)

    client = SklearnSVMClient(
        client_id=args.client_id,
        Xtr=Xtr, ytr=ytr,
        Xva=Xva, yva=yva,
        X_local=Xc, y_local=yc,
        log_dir=args.log_dir,
    )

    fl.client.start_client(server_address=args.server, client=client.to_client())

if __name__ == "__main__":
    main()
