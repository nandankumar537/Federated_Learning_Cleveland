import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

import flwr as fl

# ----- Configuration -----
FEATURE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET_COLUMN = "target"

# ----- Data utilities -----
def load_and_preprocess(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding="latin-1")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="cp1252", on_bad_lines="skip", engine="python")

    if isinstance(df.columns, pd.RangeIndex) and df.shape[1] == 14:
        df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Got columns: {list(df.columns)}")

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
    rs = np.random.RandomState(seed)
    idx = np.arange(len(yc))
    cls0 = idx[yc == 0]
    cls1 = idx[yc == 1]
    rs.shuffle(cls0); rs.shuffle(cls1)
    n0_val = max(1, int(len(cls0) * val_ratio))
    n1_val = max(1, int(len(cls1) * val_ratio))
    val_idx = np.concatenate([cls0[:n0_val], cls1[:n1_val]])
    train_idx = np.concatenate([cls0[n0_val:], cls1[n1_val:]])
    rs.shuffle(train_idx); rs.shuffle(val_idx)
    return (Xc[train_idx], yc[train_idx]), (Xc[val_idx], yc[val_idx])

# ----- Training / Eval -----
def evaluate_metrics_sklearn(model, X, y) -> Dict[str, float]:
    p = model.predict(X)
    acc = accuracy_score(y, p)
    prec = precision_score(y, p, zero_division=0)
    rec = recall_score(y, p, zero_division=0)
    f1 = f1_score(y, p, zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def mean_cv_accuracy_local(X_local: np.ndarray, y_local: np.ndarray, folds: int = 5) -> float:
    skf = StratifiedKFold(n_splits=min(folds, len(np.unique(y_local)) if len(y_local) >= folds else 2), shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf.split(X_local, y_local):
        Xtr, Xva = X_local[tr_idx], X_local[va_idx]
        ytr, yva = y_local[tr_idx], y_local[va_idx]
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1, max_iter=1000,
            random_state=42
        )
        model.fit(Xtr, ytr)
        yhat = model.predict(Xva)
        accs.append(accuracy_score(yva, yhat))
    return float(np.mean(accs)) if accs else 0.0

# ----- Flower client for sklearn -----
class SklearnLRClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_local: np.ndarray,
        y_local: np.ndarray,
        log_dir: str = ".",
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_local = X_local
        self.y_local = y_local
        self.model = LogisticRegression(
            penalty="l1", solver="liblinear", C=1, max_iter=1000,
            random_state=42
        )
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"client{client_id}_log.txt")
        # write header
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tMeanCVAccuracy\n")

    def get_parameters(self, config):
        # Return model coefficients and intercept as parameters (numpy arrays)
        # This helps Flower aggregate them
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            return [self.model.coef_, self.model.intercept_]
        else:
            # Model not yet trained, return zeros
            n_features = self.X_train.shape[1]
            return [np.zeros((1, n_features)), np.zeros(1)]

    def set_parameters(self, parameters: List[np.ndarray]):
        # Set model coefficients and intercept
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.classes_ = np.array([0, 1])  # Needed for sklearn's predict

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", -1))
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config={}), len(self.y_train), {"server_round": server_round}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        server_round = int(config.get("server_round", -1))
        base_metrics = evaluate_metrics_sklearn(self.model, self.X_val, self.y_val)
        mean_cv_acc = mean_cv_accuracy_local(self.X_local, self.y_local, folds=5)
        metrics = {
            "accuracy": float(base_metrics["accuracy"]),
            "precision": float(base_metrics["precision"]),
            "recall": float(base_metrics["recall"]),
            "f1": float(base_metrics["f1"]),
            "mean_cv_accuracy": float(mean_cv_acc),
            "server_round": server_round,
            "client_id": self.client_id,
        }
        # Append to client log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{server_round}\t"
                f"{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t"
                f"{metrics['f1']:.4f}\t{metrics['mean_cv_accuracy']:.4f}\n"
            )
        # For Flower, return a dummy float as loss (sklearn doesn't provide loss by default)
        return 0.0, len(self.y_val), metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cleveland.csv", help="Path to cleveland dataset CSV")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Server address host:port")
    parser.add_argument("--num_clients", type=int, default=2, help="Total number of clients in the federation")
    parser.add_argument("--client_id", type=int, required=True, help="Client id in [0..num_clients-1]")
    parser.add_argument("--log_dir", type=str, default=".", help="Directory to write client logs")
    args = parser.parse_args()

    # Load and preprocess full dataset
    X, y = load_and_preprocess(args.data)

    # Partition this client's shard
    Xc, yc = partition_data(X, y, num_clients=args.num_clients, client_id=args.client_id, seed=42)

    # Local train/val split
    (Xtr, ytr), (Xva, yva) = train_val_split(Xc, yc, val_ratio=0.2, seed=123)

    client = SklearnLRClient(
        client_id=args.client_id,
        X_train=Xtr,
        y_train=ytr,
        X_val=Xva,
        y_val=yva,
        X_local=Xc,
        y_local=yc,
        log_dir=args.log_dir,
    )

    fl.client.start_client(server_address=args.server, client=client.to_client())

if __name__ == "__main__":

    main()
