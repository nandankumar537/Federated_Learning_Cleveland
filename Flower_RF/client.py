# client_rf_dp.py
import argparse
import os
import time
import math
import numpy as np
import pandas as pd
import psutil
from typing import Tuple, Dict, List

import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

# ----- Config -----
FEATURE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET_COLUMN = "target"

# -----------------------
# Data utilities
# -----------------------
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
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X, y

def partition_data(X: np.ndarray, y: np.ndarray, num_clients: int, client_id: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    for i, (_, idx_client) in enumerate(skf.split(X, y)):
        if i == client_id:
            return X[idx_client], y[idx_client]
    raise ValueError("Invalid client_id or partitioning failure")

def train_val_split(Xc: np.ndarray, yc: np.ndarray, val_ratio: float = 0.2, seed: int = 123):
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
        clf = RandomForestClassifier(max_depth=5, n_estimators=50, min_samples_split=5, min_samples_leaf=1, random_state=42)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xva)
        accs.append(accuracy_score(yva, yhat))
    return float(np.mean(accs)) if accs else 0.0

# -----------------------
# System metrics
# -----------------------
def get_client_disk_usage(log_dir="."):
    total_bytes = 0
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, file))
            except Exception:
                pass
    return total_bytes / (1024 ** 2)  # MB

def sample_system_metrics(log_dir, samples=10, interval=0.05):
    cpu_samples = []
    ram_samples = []
    disk_samples = []
    process = psutil.Process(os.getpid())
    for _ in range(samples):
        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples.append(process.memory_info().rss / (1024 ** 2))
        disk_samples.append(get_client_disk_usage(log_dir))
        time.sleep(interval)
    avg_cpu = sum(cpu_samples) / len(cpu_samples)
    avg_ram = sum(ram_samples) / len(ram_samples)
    avg_disk = sum(disk_samples) / len(disk_samples)
    return avg_cpu, avg_ram, avg_disk

# -----------------------
# DP helpers (same as logistic version, but applied to flattened prediction vector)
# -----------------------
def l2_clip(v: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= clip_norm or norm == 0.0:
        return v
    return (v / norm) * clip_norm

def gaussian_noise(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return v
    return v + np.random.normal(loc=0.0, scale=sigma, size=v.shape)

def compute_sigma_from_epsilon_delta(clip_norm: float, eps: float, delta: float) -> float:
    if eps <= 0 or delta <= 0:
        raise ValueError("eps and delta must be > 0")
    return (clip_norm * math.sqrt(2.0 * math.log(1.25 / delta))) / eps

# -----------------------
# Client implementation
# -----------------------
class RFDPClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_local: np.ndarray,
        y_local: np.ndarray,
        X_public: np.ndarray,
        dp_enabled: bool,
        dp_epsilon: float,
        dp_delta: float,
        clip_norm: float,
        log_dir: str = ".",
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_local = X_local
        self.y_local = y_local
        self.X_public = X_public  # may be None -> no prediction-aggregation mode
        self.dp_enabled = dp_enabled
        self.dp_epsilon = float(dp_epsilon)
        self.dp_delta = float(dp_delta)
        self.clip_norm = float(clip_norm)
        self.model = RandomForestClassifier(max_depth=5, n_estimators=50, min_samples_split=5, min_samples_leaf=1, random_state=42)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f"client{client_id}_log.txt")
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tMeanCVAccuracy\tExecTime(s)\tAvgCPU(%)\tAvgRAM(MB)\tAvgDiskUsed(MB)\n")

        # Prepare DP sigma if enabled
        if self.dp_enabled:
            # sigma computed per single release (you will need composition manually)
            self.sigma = compute_sigma_from_epsilon_delta(self.clip_norm, self.dp_epsilon, self.dp_delta)
        else:
            self.sigma = 0.0

        # placeholder for last global prediction vector received from server
        self.prev_pred_vector = None
        if self.X_public is None:
            # no public data mode: prev_pred_vector stays None and we fallback to no-params mode
            pass

    def _preds_to_vector(self, preds: np.ndarray) -> np.ndarray:
        # preds shape (n_public, n_classes) -> flatten
        return preds.ravel()

    def _vector_to_preds(self, vec: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        return vec.reshape(shape)

    def get_parameters(self, config):
        # When using public-data prediction-aggregation mode, return flattened prediction vector
        if self.X_public is None or self.prev_pred_vector is None:
            return []  # no parameter exchange
        return [self.prev_pred_vector.copy()]

    def set_parameters(self, parameters):
        # Server sends flattened prediction vector as single numpy array
        if not parameters:
            self.prev_pred_vector = None
            return
        vec = np.asarray(parameters[0], dtype=float).ravel()
        self.prev_pred_vector = vec

    def fit(self, parameters, config):
        t0 = time.time()
        avg_cpu, avg_ram, avg_disk = sample_system_metrics(self.log_dir, samples=10, interval=0.05)
        server_round = int(config.get("server_round", -1))

        # Set incoming pred-vector if present
        self.set_parameters(parameters)

        # Train locally
        self.model.fit(self.X_train, self.y_train)

        exec_time = time.time() - t0

        # Evaluate locally (on validation)
        p = self.model.predict(self.X_val)
        acc = accuracy_score(self.y_val, p)
        prec = precision_score(self.y_val, p, zero_division=0)
        rec = recall_score(self.y_val, p, zero_division=0)
        f1 = f1_score(self.y_val, p, zero_division=0)
        mean_cv = mean_cv_accuracy_local(self.X_local, self.y_local, folds=5)

        # If no public data provided, we cannot participate in param aggregation; return empty params
        if self.X_public is None:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{server_round}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{mean_cv:.4f}\t{exec_time:.4f}\t{avg_cpu:.2f}\t{avg_ram:.2f}\t{avg_disk:.2f}\n")
            return [], len(self.y_train), {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "mean_cv_accuracy": float(mean_cv),
                "server_round": server_round,
                "client_id": self.client_id,
                "exec_time": exec_time,
                "avg_cpu": avg_cpu,
                "avg_ram": avg_ram,
                "avg_disk": avg_disk,
            }

        # Compute predictions on public set
        preds = self.model.predict_proba(self.X_public)  # shape (n_public, n_classes)
        pred_vec = self._preds_to_vector(preds)

        # If prev_pred_vector is not set, create zeros vector of same shape (server likely initialized zeros)
        if self.prev_pred_vector is None:
            self.prev_pred_vector = np.zeros_like(pred_vec)

        # compute delta
        delta = pred_vec - self.prev_pred_vector

        # apply clipping and noise
        if self.dp_enabled:
            delta_clipped = l2_clip(delta, self.clip_norm)
            delta_noisy = gaussian_noise(delta_clipped, self.sigma)
            vec_to_send = self.prev_pred_vector + delta_noisy
        else:
            vec_to_send = pred_vec

        # set prev_pred_vector to vec_to_send for local consistency (so evaluate uses same)
        self.prev_pred_vector = vec_to_send.copy()

        # Log everything
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{server_round}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{mean_cv:.4f}\t{exec_time:.4f}\t{avg_cpu:.2f}\t{avg_ram:.2f}\t{avg_disk:.2f}\n")

        # Return flattened prediction vector as the single parameter array
        return [vec_to_send.astype(float)], len(self.y_train), {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_cv_accuracy": float(mean_cv),
            "server_round": server_round,
            "client_id": self.client_id,
            "exec_time": exec_time,
            "avg_cpu": avg_cpu,
            "avg_ram": avg_ram,
            "avg_disk": avg_disk,
        }

    def evaluate(self, parameters, config):
        # set incoming pred-vector if provided
        self.set_parameters(parameters)
        # Evaluate current RF on local validation
        p = self.model.predict(self.X_val)
        acc = accuracy_score(self.y_val, p)
        prec = precision_score(self.y_val, p, zero_division=0)
        rec = recall_score(self.y_val, p, zero_division=0)
        f1 = f1_score(self.y_val, p, zero_division=0)
        mean_cv = mean_cv_accuracy_local(self.X_local, self.y_local, folds=5)
        return 0.0, len(self.y_val), {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "mean_cv_accuracy": float(mean_cv),
        }

# -----------------------
# CLI and entrypoint
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cleveland.csv")
    parser.add_argument("--public_data", type=str, default=None, help="Path to shared public data (numpy .npy) used to compute predict_proba")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--dp_enabled", action="store_true")
    parser.add_argument("--dp_epsilon", type=float, default=1.0)
    parser.add_argument("--dp_delta", type=float, default=1e-5)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    return parser.parse_args()

def main():
    args = parse_args()

    X, y = load_and_preprocess(args.data)
    Xc, yc = partition_data(X, y, num_clients=args.num_clients, client_id=args.client_id, seed=42)
    (Xtr, ytr), (Xva, yva) = train_val_split(Xc, yc, val_ratio=0.2, seed=123)

    X_public = None
    if args.public_data:
        X_public = np.load(args.public_data)  # must be shape (n_public, n_features)

    client = RFDPClient(
        client_id=args.client_id,
        X_train=Xtr,
        y_train=ytr,
        X_val=Xva,
        y_val=yva,
        X_local=Xc,
        y_local=yc,
        X_public=X_public,
        dp_enabled=args.dp_enabled,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        clip_norm=args.clip_norm,
        log_dir=args.log_dir,
    )

    fl.client.start_client(server_address=args.server, client=client)

if __name__ == "__main__":
    import math
    main()
