"""
Delayed Feedback CVR Estimation Experiment
Implements: Naive, DFM, ES-DFM, FDAM (Weibull-based flexible distribution with online label correction)
Dataset: Criteo Conversion Logs
"""
import numpy as np
import time
import sys
from sklearn.metrics import roc_auc_score

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_dataset/data.txt"
MAX_ROWS = 16_000_000  # 全量数据（原为 500_000 快速验证）
HIDDEN_DIM = 64
EPOCHS = 3
BATCH_SIZE = 512
LR = 0.01
SEED = 42
np.random.seed(SEED)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(path, max_rows=MAX_ROWS):
    """Load Criteo dataset. Columns: click_ts, conv_ts, int*8, cat*9"""
    click_ts_list = []
    conv_ts_list = []
    int_feats = []
    cat_feats = []

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 19:
                continue
            click_ts = int(parts[0]) if parts[0].strip() else 0
            conv_ts_str = parts[1].strip()
            conv_ts = int(conv_ts_str) if conv_ts_str else -1  # -1 = no conversion

            ints = []
            for j in range(2, 10):
                v = parts[j].strip()
                try:
                    ints.append(float(v) if v else 0.0)
                except ValueError:
                    ints.append(0.0)

            cats = []
            for j in range(10, 19):
                v = parts[j].strip() if j < len(parts) else ''
                cats.append(v)

            click_ts_list.append(click_ts)
            conv_ts_list.append(conv_ts)
            int_feats.append(ints)
            cat_feats.append(cats)

    return (np.array(click_ts_list, dtype=np.int64),
            np.array(conv_ts_list, dtype=np.int64),
            np.array(int_feats, dtype=np.float32),
            cat_feats)


def encode_cats(cat_feats_list):
    """Label-encode categorical features using dict hash."""
    n = len(cat_feats_list)
    n_cat = len(cat_feats_list[0]) if n > 0 else 9
    result = np.zeros((n, n_cat), dtype=np.float32)
    encoders = [{} for _ in range(n_cat)]
    for i, row in enumerate(cat_feats_list):
        for j, val in enumerate(row):
            if val not in encoders[j]:
                encoders[j][val] = len(encoders[j])
            result[i, j] = encoders[j][val]
    return result


def build_features(int_feats, cat_feats_list):
    cat_encoded = encode_cats(cat_feats_list)
    X = np.concatenate([int_feats, cat_encoded], axis=1)
    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    return X.astype(np.float32)


# ─── MLP (pure numpy) ─────────────────────────────────────────────────────────

class MLP:
    """2-layer MLP with ReLU, sigmoid output."""

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = (np.random.randn(input_dim, hidden_dim) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, 1) * scale2).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = 1.0 / (1.0 + np.exp(-np.clip(self.z2, -30, 30)))
        return self.out

    def backward(self, y, weights=None):
        """BCE loss backward. weights: sample weights (optional)."""
        n = len(y)
        y = y.reshape(-1, 1)
        if weights is None:
            weights = np.ones(n, dtype=np.float32)
        weights = weights.reshape(-1, 1)

        # d_loss / d_z2
        d_out = (self.out - y) * weights / n
        dW2 = self.a1.T @ d_out
        db2 = d_out.sum(axis=0)

        d_a1 = d_out @ self.W2.T
        d_z1 = d_a1 * (self.z1 > 0).astype(np.float32)
        dW1 = self.X.T @ d_z1
        db1 = d_z1.sum(axis=0)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr=LR):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2 + self.b2
        return (1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30)))).ravel()


def train_mlp(model, X, y, weights=None, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    n = len(X)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = idx[start:end]
            Xb = X[bi]
            yb = y[bi].astype(np.float32)
            wb = weights[bi] if weights is not None else None
            model.forward(Xb)
            grads = model.backward(yb, wb)
            model.update(*grads, lr=lr)


# ─── Delay Distribution Helpers ───────────────────────────────────────────────

def compute_delay(click_ts, conv_ts):
    """Delay in seconds. -1 for non-conversions."""
    delay = np.where(conv_ts > 0, (conv_ts - click_ts).astype(np.float64), -1.0)
    return delay


def fit_exponential(delays):
    """Fit exponential distribution to observed delays (MLE: lambda = 1/mean)."""
    pos = delays[delays > 0]
    if len(pos) == 0:
        return 1.0 / 86400.0
    lam = 1.0 / (pos.mean() + 1e-8)
    return lam


def exponential_survival(t, lam):
    """P(delay > t) = exp(-lambda * t)"""
    return np.exp(-lam * np.clip(t, 0, None))


def fit_weibull(delays):
    """Fit Weibull distribution to observed delays via MLE (simplified moment matching)."""
    pos = delays[delays > 0]
    if len(pos) < 2:
        return 1.0, 86400.0  # shape=1 (exponential), scale=1day
    # Method of moments for Weibull
    mu = pos.mean()
    sigma = pos.std() + 1e-8
    cv = sigma / mu  # coefficient of variation
    # Approximate shape k from CV: CV^2 ≈ Gamma(1+2/k)/Gamma(1+1/k)^2 - 1
    # Use numerical approximation
    k = max(0.1, min(10.0, (mu / sigma) ** 1.086))  # empirical approximation
    # Scale lambda: mu = lambda * Gamma(1 + 1/k)
    from math import gamma as math_gamma
    try:
        g = math_gamma(1.0 + 1.0 / k)
    except Exception:
        g = 1.0
    lam = mu / (g + 1e-8)
    return k, lam


def weibull_survival(t, k, lam):
    """P(delay > t) = exp(-(t/lambda)^k)"""
    t = np.clip(t, 0, None)
    return np.exp(-np.power(t / (lam + 1e-8), k))


# ─── Model 1: Naive ───────────────────────────────────────────────────────────

def train_naive(X_train, y_train, input_dim):
    model = MLP(input_dim)
    train_mlp(model, X_train, y_train)
    return model


# ─── Model 2: DFM (Delayed Feedback Model) ────────────────────────────────────

def train_dfm(X_train, y_train, click_ts_train, conv_ts_train, input_dim):
    """
    DFM: exponential delay model, importance weighting for non-conversions.
    For non-conversions at elapsed time e: weight = 1 / P(no conversion by e)
    """
    delays = compute_delay(click_ts_train, conv_ts_train)
    lam = fit_exponential(delays)

    # Elapsed time = current time (approx as max click_ts) - click_ts
    current_time = click_ts_train.max()
    elapsed = (current_time - click_ts_train).astype(np.float64)
    elapsed = np.maximum(elapsed, 1.0)

    # Importance weights
    weights = np.ones(len(X_train), dtype=np.float32)
    non_conv_mask = (y_train == 0)
    # P(no conversion by elapsed) = exp(-lambda * elapsed)
    p_no_conv = exponential_survival(elapsed[non_conv_mask], lam)
    # Weight = 1 / P(no conversion) — clip to avoid explosion
    weights[non_conv_mask] = np.clip(1.0 / (p_no_conv + 1e-6), 1.0, 20.0).astype(np.float32)

    model = MLP(input_dim)
    train_mlp(model, X_train, y_train, weights=weights)
    return model


# ─── Model 3: ES-DFM (Elapsed-time Sampling DFM) ─────────────────────────────

def train_esdfm(X_train, y_train, click_ts_train, conv_ts_train, input_dim):
    """
    ES-DFM: DFM + elapsed-time sampling.
    Sample a fake observation time t_obs ~ Uniform(click_ts, current_time),
    then relabel: if conv_ts <= t_obs → converted, else → not converted.
    Use DFM weights on the relabeled data.
    """
    delays = compute_delay(click_ts_train, conv_ts_train)
    lam = fit_exponential(delays)

    current_time = click_ts_train.max()

    # Sample observation times
    elapsed_max = np.maximum((current_time - click_ts_train).astype(np.float64), 1.0)
    t_obs_frac = np.random.uniform(0, 1, size=len(X_train))
    t_obs = click_ts_train.astype(np.float64) + t_obs_frac * elapsed_max

    # Relabel based on sampled observation time
    y_relabeled = np.zeros(len(y_train), dtype=np.float32)
    converted_mask = (conv_ts_train > 0) & (conv_ts_train.astype(np.float64) <= t_obs)
    y_relabeled[converted_mask] = 1.0

    # Elapsed at observation time
    elapsed_at_obs = t_obs - click_ts_train.astype(np.float64)
    elapsed_at_obs = np.maximum(elapsed_at_obs, 1.0)

    # Importance weights (same as DFM but on relabeled)
    weights = np.ones(len(X_train), dtype=np.float32)
    non_conv_mask = (y_relabeled == 0)
    p_no_conv = exponential_survival(elapsed_at_obs[non_conv_mask], lam)
    weights[non_conv_mask] = np.clip(1.0 / (p_no_conv + 1e-6), 1.0, 20.0).astype(np.float32)

    model = MLP(input_dim)
    train_mlp(model, X_train, y_relabeled, weights=weights)
    return model


# ─── Model 4: FDAM (Flexible Distribution Approach with online label correction) ─

def train_fdam(X_train, y_train, click_ts_train, conv_ts_train, input_dim):
    """
    FDAM: Weibull delay distribution + online label correction.
    1. Fit Weibull on observed delays.
    2. For each sample, compute soft label:
       - If converted: label = 1
       - If not converted at elapsed e: soft_label = P(conversion) * P(delay > e | converted)
         = estimated_cvr * weibull_survival(e)
         We approximate estimated_cvr from initial naive pass.
    3. Online update: after each mini-batch, update Weibull params on newly "revealed" conversions.
    """
    delays = compute_delay(click_ts_train, conv_ts_train)
    k, lam = fit_weibull(delays)

    current_time = click_ts_train.max()
    elapsed = np.maximum((current_time - click_ts_train).astype(np.float64), 1.0)

    # Compute soft labels
    # For conversions: label = 1.0
    # For non-conversions: label = survival(elapsed) as correction factor
    # (soft label = P(will eventually convert | not converted yet))
    surv = weibull_survival(elapsed, k, lam)

    # Estimate base CVR from data
    base_cvr = y_train.mean()

    soft_labels = np.where(
        y_train == 1,
        1.0,
        # P(convert later | not converted by elapsed) * base_cvr
        np.clip(surv * base_cvr / (1.0 - base_cvr * (1.0 - surv) + 1e-8), 0.0, 0.5)
    ).astype(np.float32)

    # Importance weights: converted samples get higher weight
    weights = np.ones(len(X_train), dtype=np.float32)
    non_conv_mask = (y_train == 0)
    p_no_conv = weibull_survival(elapsed[non_conv_mask], k, lam)
    weights[non_conv_mask] = np.clip(1.0 / (p_no_conv + 1e-6), 1.0, 20.0).astype(np.float32)

    model = MLP(input_dim)

    # Online training: process in chunks, update Weibull params after each chunk
    n = len(X_train)
    chunk_size = max(BATCH_SIZE * 10, n // 10)

    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            bi = idx[start:end]
            Xb = X_train[bi]
            yb = soft_labels[bi]
            wb = weights[bi]
            model.forward(Xb)
            grads = model.backward(yb, wb)
            model.update(*grads, lr=LR)

            # Online Weibull update every chunk
            if (start // BATCH_SIZE) % 10 == 0 and start > 0:
                # Use delays from current chunk's conversions
                chunk_delays = delays[bi]
                new_pos = chunk_delays[chunk_delays > 0]
                if len(new_pos) > 5:
                    all_pos = delays[delays > 0]
                    # Mix old and new estimates
                    k_new, lam_new = fit_weibull(all_pos)
                    k = 0.9 * k + 0.1 * k_new
                    lam = 0.9 * lam + 0.1 * lam_new
                    # Recompute soft labels for remaining samples
                    surv = weibull_survival(elapsed, k, lam)
                    soft_labels = np.where(
                        y_train == 1,
                        1.0,
                        np.clip(surv * base_cvr / (1.0 - base_cvr * (1.0 - surv) + 1e-8), 0.0, 0.5)
                    ).astype(np.float32)

    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading data...", flush=True)

    click_ts, conv_ts, int_feats, cat_feats_list = load_data(DATA_PATH, MAX_ROWS)
    n = len(click_ts)
    print(f"Loaded {n} rows in {time.time()-t0:.1f}s", flush=True)

    # Build features
    print("Building features...", flush=True)
    X = build_features(int_feats, cat_feats_list)
    y = (conv_ts > 0).astype(np.float32)
    print(f"CVR: {y.mean():.4f}, features: {X.shape[1]}", flush=True)

    # Chronological split: 60/20/20
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    n_test = n - n_train - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    click_ts_train, conv_ts_train = click_ts[:n_train], conv_ts[:n_train]

    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]

    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}", flush=True)
    print(f"Test CVR: {y_test.mean():.4f}", flush=True)

    input_dim = X.shape[1]

    # ── Naive ──
    print("Training Naive...", flush=True)
    t1 = time.time()
    naive_model = train_naive(X_train, y_train, input_dim)
    naive_preds = naive_model.predict(X_test)
    naive_auc = roc_auc_score(y_test, naive_preds)
    print(f"  Naive AUC: {naive_auc:.4f} ({time.time()-t1:.1f}s)", flush=True)

    # ── DFM ──
    print("Training DFM...", flush=True)
    t1 = time.time()
    dfm_model = train_dfm(X_train, y_train, click_ts_train, conv_ts_train, input_dim)
    dfm_preds = dfm_model.predict(X_test)
    dfm_auc = roc_auc_score(y_test, dfm_preds)
    print(f"  DFM AUC: {dfm_auc:.4f} ({time.time()-t1:.1f}s)", flush=True)

    # ── ES-DFM ──
    print("Training ES-DFM...", flush=True)
    t1 = time.time()
    esdfm_model = train_esdfm(X_train, y_train, click_ts_train, conv_ts_train, input_dim)
    esdfm_preds = esdfm_model.predict(X_test)
    esdfm_auc = roc_auc_score(y_test, esdfm_preds)
    print(f"  ES-DFM AUC: {esdfm_auc:.4f} ({time.time()-t1:.1f}s)", flush=True)

    # ── FDAM ──
    print("Training FDAM...", flush=True)
    t1 = time.time()
    fdam_model = train_fdam(X_train, y_train, click_ts_train, conv_ts_train, input_dim)
    fdam_preds = fdam_model.predict(X_test)
    fdam_auc = roc_auc_score(y_test, fdam_preds)
    print(f"  FDAM AUC: {fdam_auc:.4f} ({time.time()-t1:.1f}s)", flush=True)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s", flush=True)

    # ── Output metrics ──
    print(f"\nprimary_metric: {fdam_auc:.6f}")
    print(f"naive_auc: {naive_auc:.6f}")
    print(f"dfm_auc: {dfm_auc:.6f}")
    print(f"esdfm_auc: {esdfm_auc:.6f}")
    print(f"fdam_auc: {fdam_auc:.6f}")
    print(f"fdam_vs_naive_delta: {fdam_auc - naive_auc:.6f}")
    print(f"fdam_vs_dfm_delta: {fdam_auc - dfm_auc:.6f}")
    print(f"fdam_vs_esdfm_delta: {fdam_auc - esdfm_auc:.6f}")


if __name__ == "__main__":
    main()
