"""
datasets/synthetic.py - 合成多任务数据集

生成与 Ali-CCP 格式一致的合成数据，用于快速调试和基线验证。
"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Optional, Tuple

from .base import BaseMultiTaskDataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 特征定义
# ─────────────────────────────────────────────────────────────

SPARSE_FEATURES = ["user_id", "item_id", "category_id", "hour", "day_of_week"]
DENSE_FEATURES  = ["user_click_count", "item_ctr"]


# ─────────────────────────────────────────────────────────────
# 合成数据生成
# ─────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_samples: int = 500_000,
    n_users: int = 10_000,
    n_items: int = 50_000,
    n_categories: int = 100,
    seed: int = 42,
):
    """
    生成合成多任务数据。

    Returns
    -------
    sparse_arrays : dict {feat_name: np.ndarray int64}
    dense_array   : np.ndarray float32 (n, len(DENSE_FEATURES))
    labels        : np.ndarray float32 (n, 3)  [ctr, cvr, ctcvr]
    sparse_vocab  : dict {feat_name: vocab_size}
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    user_ids     = rng.integers(0, n_users,     size=n)
    item_ids     = rng.integers(0, n_items,     size=n)
    cat_ids      = rng.integers(0, n_categories, size=n)
    hour         = rng.integers(0, 24, size=n)
    day_of_week  = rng.integers(0, 7,  size=n)
    click_cnt    = rng.integers(0, 200, size=n).astype(float)
    item_ctr_val = rng.uniform(0, 1, size=n)

    # CTR
    ctr_logit = (
        -3.0
        + 0.01 * (click_cnt / 200.0)
        + 2.0  * item_ctr_val
        + 0.5  * (cat_ids % 10 / 10.0)
        + 0.3  * (hour / 24.0)
        + rng.normal(0, 0.5, size=n)
    )
    ctr_prob  = 1 / (1 + np.exp(-ctr_logit))
    ctr_label = (rng.uniform(size=n) < ctr_prob).astype(np.float32)

    # CVR
    cvr_logit = (
        -1.5
        + 0.02 * (click_cnt / 200.0)
        + 1.5  * item_ctr_val
        + 0.4  * (cat_ids % 5 / 5.0)
        + rng.normal(0, 0.5, size=n)
    )
    cvr_prob  = 1 / (1 + np.exp(-cvr_logit))
    cvr_label = (rng.uniform(size=n) < cvr_prob).astype(np.float32)
    ctcvr_label = (ctr_label * cvr_label).astype(np.float32)

    logger.info(
        "合成数据 | n=%d | CTR=%.3f | CVR=%.3f | CTCVR=%.3f",
        n, ctr_label.mean(), cvr_label.mean(), ctcvr_label.mean(),
    )

    sparse_arrays = {
        "user_id":     user_ids.astype(np.int64),
        "item_id":     item_ids.astype(np.int64),
        "category_id": cat_ids.astype(np.int64),
        "hour":        hour.astype(np.int64),
        "day_of_week": day_of_week.astype(np.int64),
    }
    sparse_vocab = {
        "user_id":     n_users,
        "item_id":     n_items,
        "category_id": n_categories,
        "hour":        24,
        "day_of_week": 7,
    }

    # 标准化数值特征
    click_cnt_norm = ((click_cnt - click_cnt.mean()) / (click_cnt.std() + 1e-8)).astype(np.float32)
    item_ctr_norm  = ((item_ctr_val - item_ctr_val.mean()) / (item_ctr_val.std() + 1e-8)).astype(np.float32)
    dense_array = np.stack([click_cnt_norm, item_ctr_norm], axis=1)

    labels = np.stack([ctr_label, cvr_label, ctcvr_label], axis=1)

    return sparse_arrays, dense_array, labels, sparse_vocab


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class _SyntheticTorchDataset(Dataset):
    def __init__(self, sparse_arrays, dense_array, labels):
        self.sparse_tensors = {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in sparse_arrays.items()
        }
        self.dense_tensor = torch.tensor(dense_array, dtype=torch.float32)
        self.ctr_labels   = torch.tensor(labels[:, 0], dtype=torch.float32)
        self.cvr_labels   = torch.tensor(labels[:, 1], dtype=torch.float32)
        self.ctcvr_labels = torch.tensor(labels[:, 2], dtype=torch.float32)
        self.n = len(labels)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        features = {k: v[idx] for k, v in self.sparse_tensors.items()}
        features["dense"] = self.dense_tensor[idx]
        return features, self.ctr_labels[idx], self.cvr_labels[idx], self.ctcvr_labels[idx]


# ─────────────────────────────────────────────────────────────
# Dataset 加载器
# ─────────────────────────────────────────────────────────────

class SyntheticDataset(BaseMultiTaskDataset):
    """合成多任务数据集加载器"""

    sparse_feats = SPARSE_FEATURES
    dense_feats  = DENSE_FEATURES
    dense_dim    = len(DENSE_FEATURES)

    def get_dataloaders(self, config, val_ratio: float = 0.1):
        n = config.sample_size if config.sample_size else 500_000
        sparse_arrays, dense_array, labels, sparse_vocab = generate_synthetic_data(
            n_samples=n, seed=config.seed
        )

        ds = _SyntheticTorchDataset(sparse_arrays, dense_array, labels)

        n_val   = int(len(ds) * val_ratio)
        n_train = len(ds) - n_val
        train_sub, val_sub = random_split(
            ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )

        # test：重新生成一批（不同 seed）
        n_test = max(1, n // 5)
        sp_t, d_t, l_t, _ = generate_synthetic_data(n_samples=n_test, seed=config.seed + 1)
        test_ds = _SyntheticTorchDataset(sp_t, d_t, l_t)

        train_loader = DataLoader(train_sub, batch_size=config.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_sub,   batch_size=config.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,   batch_size=config.batch_size,
                                  shuffle=False, num_workers=0)

        feature_info = {
            "sparse_vocab": sparse_vocab,
            "sparse_feats": SPARSE_FEATURES,
            "dense_feats":  DENSE_FEATURES,
            "dense_dim":    len(DENSE_FEATURES),
        }
        return train_loader, val_loader, test_loader, feature_info
