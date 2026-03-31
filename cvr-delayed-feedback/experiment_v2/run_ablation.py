#!/usr/bin/env python3
"""
Ablation study for FDAM + DEFER bug fix.

Tasks:
1. Fix DEFER data leakage bug and re-run
2. FDAM ablation variants:
   - FDAM-noWeibull: replace Weibull with exponential delay (like DFM)
   - FDAM-noAux: remove tn/dp pretrain, set importance weights to 1
   - FDAM-noSoftLabel: remove soft label, use hard 0 for non-converted

Usage:
    python run_ablation.py <method> [sample_ratio]

Methods:
    DEFER-fixed        -- DEFER with data leakage fixed
    FDAM-noWeibull     -- FDAM without Weibull (use exponential)
    FDAM-noAux         -- FDAM without tn/dp auxiliary model
    FDAM-noSoftLabel   -- FDAM without soft label correction
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict

from config import Config, SECONDS_AN_HOUR
from data import load_criteo_data, DataDF
from models import get_model
from loss import get_loss_fn, stable_log1pex, fdam_loss
from metrics import evaluate
from trainer import CriteoDataset, collate_fn


# =============================================================================
# DEFER Fix: remove data leakage in duplicate negatives
# =============================================================================

def add_defer_fixed(data: DataDF, cut_size: int, attr_win: int) -> DataDF:
    """
    Fixed DEFER data processing.

    Bug in original: duplicated negatives use oracle labels (self.labels[~label_mask]),
    which leaks future conversion info.

    Fix: duplicated negatives always get label=0 (they are observed as non-converted
    at click_ts + attr_win, so label should be their observed status = 0).
    """
    inw_mask = np.logical_and(data.pay_ts > 0, data.pay_ts - data.click_ts <= cut_size)
    label_mask = np.logical_and(data.pay_ts > 0, data.labels > 0)
    neg_mask = ~label_mask

    x = data.x.copy()

    x_combined = __import__('pandas').concat([
        x[inw_mask],
        x[~inw_mask],
        x[label_mask],
        x[neg_mask],
    ], ignore_index=True)

    sample_ts = np.concatenate([
        data.click_ts[inw_mask],
        data.click_ts[~inw_mask],
        data.pay_ts[label_mask],
        data.click_ts[neg_mask] + attr_win,   # same as original
    ])

    click_ts = np.concatenate([
        data.click_ts[inw_mask],
        data.click_ts[~inw_mask],
        data.click_ts[label_mask],
        data.click_ts[neg_mask],
    ])

    pay_ts = np.concatenate([
        data.pay_ts[inw_mask],
        data.pay_ts[~inw_mask],
        data.pay_ts[label_mask],
        data.pay_ts[neg_mask],
    ])

    labels = np.concatenate([
        np.ones(np.sum(inw_mask), dtype=np.int32),
        np.zeros(np.sum(~inw_mask), dtype=np.int32),
        data.labels[label_mask],
        np.zeros(np.sum(neg_mask), dtype=np.int32),  # FIX: always 0, no oracle leak
    ])

    idx = np.argsort(sample_ts)
    return DataDF(
        x_combined.iloc[idx].reset_index(drop=True),
        click_ts[idx],
        pay_ts[idx],
        sample_ts[idx],
        labels[idx],
    )


# =============================================================================
# FDAM Ablation Loss Functions
# =============================================================================

def fdam_no_weibull_loss(outputs: Dict[str, torch.Tensor],
                         targets: Dict[str, torch.Tensor],
                         eps: float = 1e-7) -> torch.Tensor:
    """
    FDAM-noWeibull: replace Weibull survival with exponential survival.
    S(e) = exp(-lambda * e)  (exponential, like DFM)
    """
    x = outputs["logits"].squeeze(-1)
    log_lamb_raw = outputs.get("log_lamb", torch.zeros_like(x)).squeeze(-1)
    tn_logits = outputs["tn_logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)

    z = targets["label"].float()
    elapsed = targets.get("elapsed", torch.ones_like(z)).float()

    tn_prob = torch.sigmoid(tn_logits.detach()).clamp(eps, 1 - eps)
    dp_prob = torch.sigmoid(dp_logits.detach()).clamp(eps, 1 - eps)

    # Exponential survival (no Weibull shape parameter k)
    lamb = F.softplus(log_lamb_raw) + 1.0
    survival = torch.exp(-lamb * elapsed).clamp(0.0, 1.0)

    pos_weight = 1 + dp_prob
    neg_weight = tn_prob * (1 + dp_prob)

    pos_loss = F.binary_cross_entropy_with_logits(x, torch.ones_like(z), reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(x, survival.detach(), reduction='none')

    loss = pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z)
    return loss.mean()


def fdam_no_aux_loss(outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor],
                     eps: float = 1e-7) -> torch.Tensor:
    """
    FDAM-noAux: remove tn/dp importance weights (set to 1).
    Keep Weibull soft label, but no auxiliary pretrain model.
    """
    x = outputs["logits"].squeeze(-1)
    log_k_raw = outputs.get("log_k", torch.zeros_like(x)).squeeze(-1)
    log_lamb_raw = outputs.get("log_lamb", torch.zeros_like(x)).squeeze(-1)

    z = targets["label"].float()
    elapsed = targets.get("elapsed", torch.ones_like(z)).float()

    k = F.softplus(log_k_raw) + 0.1
    lamb = F.softplus(log_lamb_raw) + 1.0
    survival = torch.exp(-torch.pow(elapsed / (lamb + eps), k)).clamp(0.0, 1.0)

    # No importance weights (all = 1)
    pos_loss = F.binary_cross_entropy_with_logits(x, torch.ones_like(z), reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(x, survival.detach(), reduction='none')

    loss = pos_loss * z + neg_loss * (1 - z)
    return loss.mean()


def fdam_no_softlabel_loss(outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor],
                           eps: float = 1e-7) -> torch.Tensor:
    """
    FDAM-noSoftLabel: remove soft label, use hard 0 for non-converted.
    Keep Weibull + tn/dp importance weights, but no label correction.
    """
    x = outputs["logits"].squeeze(-1)
    tn_logits = outputs["tn_logits"].squeeze(-1)
    dp_logits = outputs["dp_logits"].squeeze(-1)

    z = targets["label"].float()

    tn_prob = torch.sigmoid(tn_logits.detach()).clamp(eps, 1 - eps)
    dp_prob = torch.sigmoid(dp_logits.detach()).clamp(eps, 1 - eps)

    pos_weight = 1 + dp_prob
    neg_weight = tn_prob * (1 + dp_prob)

    # Hard label: non-converted = 0 (no soft label)
    pos_loss = F.binary_cross_entropy_with_logits(x, torch.ones_like(z), reduction='none')
    neg_loss = F.binary_cross_entropy_with_logits(x, torch.zeros_like(z), reduction='none')

    loss = pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z)
    return loss.mean()


ABLATION_LOSS_MAP = {
    "FDAM-noWeibull": fdam_no_weibull_loss,
    "FDAM-noAux": fdam_no_aux_loss,
    "FDAM-noSoftLabel": fdam_no_softlabel_loss,
}


# =============================================================================
# Main Runner
# =============================================================================

def run_method(method: str, sample_ratio: float = 0.03):
    print(f"\n{'='*60}")
    print(f"Running: {method}  (sample_ratio={sample_ratio*100:.0f}%)")
    print(f"{'='*60}")
    start_time = time.time()

    config = Config()
    device = 'cpu'

    # Load data
    print("Loading data...")
    features, click_ts, pay_ts = load_criteo_data(config.data.data_path)

    n_samples = int(len(features) * sample_ratio)
    print(f"Using {n_samples} samples ({sample_ratio*100:.0f}% of {len(features)})")
    np.random.seed(42)
    idx = np.random.choice(len(features), n_samples, replace=False)
    idx.sort()
    features = features.iloc[idx].reset_index(drop=True)
    click_ts = click_ts[idx]
    pay_ts = pay_ts[idx]

    data = DataDF.from_raw(features, click_ts, pay_ts, config.data.attribution_window)
    print(f"Data: {len(data)} samples, pos_rate={data.labels.mean()*100:.2f}%")

    ob_win = config.data.observation_window
    attr_win = config.data.attribution_window

    # Prepare train data
    if method == "DEFER-fixed":
        train_data = add_defer_fixed(data, ob_win, attr_win)
    elif method in ["FDAM-noWeibull", "FDAM-noSoftLabel"]:
        train_data = data.add_esdfm_cut_fake_neg(ob_win)
    elif method == "FDAM-noAux":
        train_data = data.add_esdfm_cut_fake_neg(ob_win)
    else:
        raise ValueError(f"Unknown ablation method: {method}")

    print(f"Train data: {len(train_data)} samples, pos_rate={train_data.labels.mean()*100:.2f}%")

    # Pretrain auxiliary models if needed
    aux_models = {}
    if method in ["DEFER-fixed", "FDAM-noWeibull", "FDAM-noSoftLabel"]:
        print("\nPretraining tn/dp model...")
        pretrain_data = data.construct_tn_dp_data(ob_win, attr_win)
        pretrain_dataset = CriteoDataset(pretrain_data, config.data.cat_bin_sizes)
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )

        if method == "DEFER-fixed":
            # DEFER uses dp-only pretrain
            aux_model = get_model("MLP_dp",
                                  hidden_dims=config.model.hidden_dims,
                                  embed_dim=config.data.embed_dim,
                                  use_bn=config.model.use_batch_norm).to(device)
            pretrain_loss_fn = get_loss_fn("pretrain_dp")
            aux_key = "defer"
        else:
            # FDAM variants use tn/dp pretrain
            aux_model = get_model("MLP_tn_dp",
                                  hidden_dims=config.model.hidden_dims,
                                  embed_dim=config.data.embed_dim,
                                  use_bn=config.model.use_batch_norm).to(device)
            pretrain_loss_fn = get_loss_fn("pretrain_tn_dp")
            aux_key = "esdfm"

        optimizer = torch.optim.Adam(aux_model.parameters(), lr=config.train.learning_rate)
        aux_model.train()
        for batch in tqdm(pretrain_loader, desc="Pretrain"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = aux_model(batch["num_features"], batch["cat_features"])
            loss = pretrain_loss_fn(outputs, batch)
            loss.backward()
            optimizer.step()

        aux_model.eval()
        aux_models[aux_key] = aux_model
        print("Pretrain done")

    # Create main model
    if method == "DEFER-fixed":
        model_type = "MLP_SIG"
    else:
        # All FDAM variants use Weibull model (even noWeibull uses log_lamb output)
        model_type = "MLP_WEIBULL_DELAY"

    model = get_model(
        model_type,
        hidden_dims=config.model.hidden_dims,
        embed_dim=config.data.embed_dim,
        use_bn=config.model.use_batch_norm,
    ).to(device)

    # Create dataloader
    dataset = CriteoDataset(train_data, config.data.cat_bin_sizes)
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Loss function
    if method == "DEFER-fixed":
        loss_fn = get_loss_fn("defer")
    else:
        loss_fn = ABLATION_LOSS_MAP[method]

    # Train
    print(f"\nTraining {method}...")
    model.train()
    total_loss = 0
    n_batches = len(loader)

    for i, batch in enumerate(tqdm(loader, desc="Training")):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
        optimizer.zero_grad()
        outputs = model(batch["num_features"], batch["cat_features"])

        # Inject auxiliary outputs
        if aux_models:
            with torch.no_grad():
                if "esdfm" in aux_models:
                    aux_out = aux_models["esdfm"](batch["num_features"], batch["cat_features"])
                    outputs["tn_logits"] = aux_out["tn_logits"]
                    outputs["dp_logits"] = aux_out["dp_logits"]
                    outputs["p_tn"] = torch.sigmoid(aux_out["tn_logits"])
                    outputs["p_dp"] = torch.sigmoid(aux_out["dp_logits"])
                if "defer" in aux_models:
                    aux_out = aux_models["defer"](batch["num_features"], batch["cat_features"])
                    outputs["dp_logits"] = aux_out["logits"]

        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 500 == 0:
            print(f"  Batch {i+1}/{n_batches}, loss={total_loss/(i+1):.4f}")

    print(f"Training loss: {total_loss/len(loader):.4f}")

    # Evaluate
    print("\nEvaluating...")
    test_dataset = CriteoDataset(data, config.data.cat_bin_sizes)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["num_features"], batch["cat_features"])
            probs = torch.sigmoid(outputs["logits"])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    metrics = evaluate(all_labels, all_probs, method)
    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"{method} Results:")
    print(f"  AUC={metrics['auc']:.4f}")
    print(f"  PR-AUC={metrics['pr_auc']:.4f}")
    print(f"  LogLoss={metrics['logloss']:.4f}")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"{'='*50}")

    output = {
        "method": method,
        "auc": float(metrics['auc']),
        "pr_auc": float(metrics['pr_auc']),
        "logloss": float(metrics['logloss']),
        "time_minutes": elapsed / 60,
        "n_samples": len(data),
        "sample_ratio": sample_ratio,
    }

    os.makedirs("results_ablation", exist_ok=True)
    fname = method.replace("/", "-")
    with open(f"results_ablation/{fname}.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to results_ablation/{fname}.json")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ablation.py <method> [sample_ratio]")
        print("Methods:")
        print("  DEFER-fixed       -- DEFER with data leakage fixed")
        print("  FDAM-noWeibull    -- FDAM without Weibull (exponential delay)")
        print("  FDAM-noAux        -- FDAM without tn/dp auxiliary model")
        print("  FDAM-noSoftLabel  -- FDAM without soft label correction")
        sys.exit(1)

    method = sys.argv[1]
    sample_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    run_method(method, sample_ratio)
