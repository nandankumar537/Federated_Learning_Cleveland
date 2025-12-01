# client_manual_dp.py
import argparse
import time
import os
import math
import numpy as np
import pandas as pd
import psutil
from typing import List, Tuple, Dict

import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------
# Feature / target names
# -----------------------
FEATURE_COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET_COLUMN = "target"


# -----------------------
# Data utils
# -----------------------
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
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
    # Stratified-ish split via shuffling then np.array_split
    idx = np.arange(len(y))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return X[splits[client_id]], y[splits[client_id]]


def train_val_split(Xc: np.ndarray, yc: np.ndarray, val_ratio: float = 0.2, seed: int = 123):
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


def evaluate_metrics_sklearn(model, X, y) -> Dict[str, float]:
    p = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
    }


def mean_cv_accuracy_local(X_local: np.ndarray, y_local: np.ndarray, folds: int = 5) -> float:
    # Safe CV: ensure at least 2 splits and not more than class counts
    n_splits = min(folds, max(2, len(np.unique(y_local))))
    skf = []
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr_idx, va_idx in skf.split(X_local, y_local):
        Xtr, Xva = X_local[tr_idx], X_local[va_idx]
        ytr, yva = y_local[tr_idx], y_local[va_idx]
        m = LogisticRegression(penalty="l1", solver="liblinear", C=1, max_iter=1000, random_state=42)
        m.fit(Xtr, ytr)
        accs.append(float(accuracy_score(yva, m.predict(Xva))))
    return float(np.mean(accs)) if accs else 0.0


# -----------------------
# Parameter utils
# -----------------------
def params_to_vector(coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return np.concatenate([coef.ravel(), np.asarray(intercept).ravel()])


def vector_to_params(vec: np.ndarray, coef_shape: Tuple[int, int], intercept_shape: Tuple[int]) -> List[np.ndarray]:
    n_coef = coef_shape[0] * coef_shape[1]
    coef = vec[:n_coef].reshape(coef_shape)
    intercept = vec[n_coef:n_coef + intercept_shape[0]].reshape(intercept_shape)
    return [coef, intercept]


def l2_clip(v: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= clip_norm:
        return v
    return (v / norm) * clip_norm


def gaussian_noise(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return v
    return v + np.random.normal(loc=0.0, scale=sigma, size=v.shape)


def compute_sigma_from_epsilon_delta(clip_norm: float, eps: float, delta: float) -> float:
    # Standard Gaussian mechanism (single release)
    if eps <= 0 or delta <= 0:
        raise ValueError("eps and delta must be > 0")
    return (clip_norm * math.sqrt(2.0 * math.log(1.25 / delta))) / eps


# -----------------------
# Flower client
# -----------------------
def make_client(
    client_id: int,
    data_path: str,
    num_clients: int,
    dp_enabled: bool,
    dp_epsilon: float,
    dp_delta: float,
    clip_norm: float,
    log_dir: str,
):
    X, y = load_and_preprocess(data_path)
    Xc, yc = partition_data(X, y, num_clients=num_clients, client_id=client_id, seed=42)
    (Xtr, ytr), (Xva, yva) = train_val_split(Xc, yc, val_ratio=0.2, seed=123)

    model = LogisticRegression(penalty="l1", solver="liblinear", C=1, max_iter=1000, random_state=42)

    # Precompute shapes
    n_features = Xtr.shape[1]
    coef_shape = (1, n_features)
    intercept_shape = (1,)

    # compute sigma for single-round mechanism
    sigma = 0.0
    if dp_enabled:
        sigma = compute_sigma_from_epsilon_delta(clip_norm, dp_epsilon, dp_delta)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"client{client_id}_log.txt")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Round\tAccuracy\tPrecision\tRecall\tF1\tMeanCVAccuracy\tExecTime\tAvgCPU\tAvgRAM\tAvgDisk\n")

    class SklearnDPClient(fl.client.NumPyClient):
        def __init__(self):
            self.client_id = client_id

        def get_parameters(self, config):
            if hasattr(model, "coef_") and hasattr(model, "intercept_"):
                return [model.coef_, model.intercept_]
            else:
                return [np.zeros(coef_shape), np.zeros(intercept_shape)]

        def set_parameters(self, parameters):
            coef, intercept = parameters[0], parameters[1]
            model.coef_ = np.array(coef)
            model.intercept_ = np.array(intercept)
            model.classes_ = np.array([0, 1])

        def fit(self, parameters, config):
            # record system stats before
            t0 = time.time()
            cpu0 = psutil.cpu_percent(interval=None)
            ram0 = psutil.virtual_memory().percent
            disk0 = psutil.disk_usage("/").percent

            # set incoming params
            self.set_parameters(parameters)
            # store incoming vector
            coef_in = np.copy(model.coef_) if hasattr(model, "coef_") else np.zeros(coef_shape)
            intercept_in = np.copy(model.intercept_) if hasattr(model, "intercept_") else np.zeros(intercept_shape)
            theta_in = params_to_vector(coef_in, intercept_in)

            # local training
            model.fit(Xtr, ytr)

            # store outgoing vector
            coef_out = model.coef_
            intercept_out = model.intercept_
            theta_out = params_to_vector(coef_out, intercept_out)

            # compute delta
            delta = theta_out - theta_in

            # apply clipping and noise if enabled
            if dp_enabled:
                delta_clipped = l2_clip(delta, clip_norm)
                delta_noisy = gaussian_noise(delta_clipped, sigma)
                theta_to_send = theta_in + delta_noisy
            else:
                theta_to_send = theta_out

            coef_send, intercept_send = vector_to_params(theta_to_send, coef_shape, intercept_shape)

            # set model to noisy params (so evaluate uses same)
            model.coef_ = coef_send
            model.intercept_ = intercept_send
            model.classes_ = np.array([0, 1])

            # system stats after
            exec_time = time.time() - t0
            cpu1 = psutil.cpu_percent(interval=None)
            ram1 = psutil.virtual_memory().percent
            disk1 = psutil.disk_usage("/").percent

            avg_cpu = (cpu0 + cpu1) / 2.0
            avg_ram = (ram0 + ram1) / 2.0
            avg_disk = (disk0 + disk1) / 2.0

            server_round = int(config.get("server_round", -1))

            # compute metrics on val
            base_metrics = evaluate_metrics_sklearn(model, Xva, yva)
            mean_cv_acc = mean_cv_accuracy_local(Xc, yc, folds=5)

            metrics = {
                "accuracy": float(base_metrics["accuracy"]),
                "precision": float(base_metrics["precision"]),
                "recall": float(base_metrics["recall"]),
                "f1": float(base_metrics["f1"]),
                "mean_cv_accuracy": float(mean_cv_acc),
                "server_round": server_round,
                "client_id": int(self.client_id),
                "exec_time": float(exec_time),
                "avg_cpu": float(avg_cpu),
                "avg_ram": float(avg_ram),
                "avg_disk": float(avg_disk),
            }

            # append to client log file
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{server_round}\t"
                    f"{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t"
                    f"{metrics['f1']:.4f}\t{metrics['mean_cv_accuracy']:.4f}\t"
                    f"{metrics['exec_time']:.4f}\t{metrics['avg_cpu']:.2f}\t{metrics['avg_ram']:.2f}\t{metrics['avg_disk']:.2f}\n"
                )

            # return noisy parameters (shaped) to server for aggregation
            return [coef_send, intercept_send], len(ytr), metrics

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            metrics = evaluate_metrics_sklearn(model, Xva, yva)
            metrics["mean_cv_accuracy"] = metrics["accuracy"]
            return 0.0, len(yva), metrics

    return SklearnDPClient()


# -----------------------
# CLI + entrypoint
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="cleveland.csv", help="CSV dataset path")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="server address host:port")
    parser.add_argument("--num_clients", type=int, default=3, help="Total number of clients")
    parser.add_argument("--client_id", type=int, required=True, help="Client id (0..num_clients-1)")
    parser.add_argument("--log_dir", type=str, default=".", help="Directory for client logs")
    parser.add_argument("--dp_enabled", action="store_true", help="Enable client-side DP (clip+noise)")
    parser.add_argument("--dp_epsilon", type=float, default=1.0, help="Per-release epsilon")
    parser.add_argument("--dp_delta", type=float, default=1e-5, help="Per-release delta")
    parser.add_argument("--clip_norm", type=float, default=1.0, help="L2 clip norm for update delta")
    return parser.parse_args()


def main():
    args = parse_args()

    client = make_client(
        client_id=args.client_id,
        data_path=args.data,
        num_clients=args.num_clients,
        dp_enabled=args.dp_enabled,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        clip_norm=args.clip_norm,
        log_dir=args.log_dir,
    )

    fl.client.start_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
