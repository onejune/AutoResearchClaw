"""
data.py - Ali-CCP 数据加载与预处理

支持：
  1. 真实 Ali-CCP 数据（common_features + sample_skeleton CSV）
  2. 自动生成合成数据（数据文件不存在时）

返回格式：(features_dict, ctr_label, cvr_label, ctcvr_label)
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 合成数据生成
# ─────────────────────────────────────────────────────────────

def generate_synthetic_data(
    n_samples: int = 500_000,
    n_users: int = 10_000,
    n_items: int = 50_000,
    n_categories: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成与 Ali-CCP 格式一致的合成数据。

    Returns
    -------
    skeleton_df : DataFrame  (sample_id, ctr_label, cvr_label, ctcvr_label)
    features_df : DataFrame  (sample_id, feature_name, feature_value)
    """
    rng = np.random.default_rng(seed)

    sample_ids = np.arange(n_samples)

    # ── 稀疏特征 ──────────────────────────────────────────
    user_ids       = rng.integers(0, n_users,     size=n_samples)
    item_ids       = rng.integers(0, n_items,     size=n_samples)
    category_ids   = rng.integers(0, n_categories, size=n_samples)

    # ── 数值特征 ──────────────────────────────────────────
    hour           = rng.integers(0, 24, size=n_samples)
    day_of_week    = rng.integers(0, 7,  size=n_samples)
    user_click_cnt = rng.integers(0, 200, size=n_samples).astype(float)
    item_ctr       = rng.uniform(0, 1, size=n_samples)

    # ── 标签生成（有真实关联，保证 AUC > 0.6）─────────────
    # CTR logit：受 user_click_cnt、item_ctr、category 影响
    ctr_logit = (
        -3.0
        + 0.01  * (user_click_cnt / 200.0)
        + 2.0   * item_ctr
        + 0.5   * (category_ids % 10 / 10.0)
        + 0.3   * (hour / 24.0)
        + rng.normal(0, 0.5, size=n_samples)
    )
    ctr_prob   = 1 / (1 + np.exp(-ctr_logit))
    ctr_label  = (rng.uniform(size=n_samples) < ctr_prob).astype(np.int8)

    # CVR logit（仅在点击样本上有意义）
    cvr_logit = (
        -1.5
        + 0.02  * (user_click_cnt / 200.0)
        + 1.5   * item_ctr
        + 0.4   * (category_ids % 5 / 5.0)
        + rng.normal(0, 0.5, size=n_samples)
    )
    cvr_prob   = 1 / (1 + np.exp(-cvr_logit))
    cvr_label  = (rng.uniform(size=n_samples) < cvr_prob).astype(np.int8)
    # CVR 只在点击后才有意义，但我们保留全部（ESMM 需要）
    ctcvr_label = (ctr_label * cvr_label).astype(np.int8)

    logger.info(
        "合成数据统计 | CTR 正样本率: %.3f | CVR(全量) 正样本率: %.3f | CTCVR 正样本率: %.3f",
        ctr_label.mean(), cvr_label.mean(), ctcvr_label.mean(),
    )

    # ── 构建 skeleton DataFrame ────────────────────────────
    skeleton_df = pd.DataFrame({
        "sample_id":   sample_ids,
        "ctr_label":   ctr_label,
        "cvr_label":   cvr_label,
        "ctcvr_label": ctcvr_label,
    })

    # ── 构建 features DataFrame（long format）─────────────
    rows = []
    feature_arrays = {
        "user_id":          user_ids,
        "item_id":          item_ids,
        "category_id":      category_ids,
        "hour":             hour,
        "day_of_week":      day_of_week,
        "user_click_count": user_click_cnt,
        "item_ctr":         item_ctr,
    }
    for feat_name, arr in feature_arrays.items():
        tmp = pd.DataFrame({
            "sample_id":     sample_ids,
            "feature_name":  feat_name,
            "feature_value": arr,
        })
        rows.append(tmp)
    features_df = pd.concat(rows, ignore_index=True)

    return skeleton_df, features_df


# ─────────────────────────────────────────────────────────────
# 特征元信息
# ─────────────────────────────────────────────────────────────

# 稀疏特征（需要 embedding）
SPARSE_FEATURES = ["user_id", "item_id", "category_id", "hour", "day_of_week"]
# 数值特征（标准化）
DENSE_FEATURES  = ["user_click_count", "item_ctr"]


# ─────────────────────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────────────────────

class AliCCPDataset(Dataset):
    """
    Ali-CCP 多任务数据集。

    Parameters
    ----------
    data_dir    : str  Ali-CCP 数据目录（含 common_features_*.csv + sample_skeleton_*.csv）
                       空字符串 → 自动生成合成数据
    split       : str  "train" | "test"
    sample_size : int | None  采样数（None=全量）
    seed        : int  随机种子
    _precomputed: dict  内部用，直接传入已处理的数据（避免重复计算）
    """

    def __init__(
        self,
        data_dir: str = "",
        split: str = "train",
        sample_size: Optional[int] = None,
        seed: int = 42,
        _precomputed: Optional[dict] = None,
    ):
        super().__init__()
        self.split = split
        self.seed  = seed

        if _precomputed is not None:
            self._load_from_precomputed(_precomputed)
            return

        # ── 加载或生成数据 ─────────────────────────────────
        skeleton_df, features_df = self._load_or_generate(data_dir, split, sample_size, seed)

        # ── 透视：long → wide ──────────────────────────────
        wide_df = self._pivot_features(features_df, skeleton_df)

        # ── 特征工程 ───────────────────────────────────────
        self._fit_transform(wide_df, skeleton_df)

    # ── 内部方法 ──────────────────────────────────────────────

    def _load_or_generate(self, data_dir, split, sample_size, seed):
        skel_path = os.path.join(data_dir, f"sample_skeleton_{split}.csv")  if data_dir else ""
        feat_path = os.path.join(data_dir, f"common_features_{split}.csv") if data_dir else ""

        if data_dir and os.path.exists(skel_path) and os.path.exists(feat_path):
            logger.info("加载真实数据: %s", data_dir)
            skeleton_df = pd.read_csv(skel_path)
            features_df = pd.read_csv(feat_path)
            # 确保列名规范
            skeleton_df.columns = [c.strip() for c in skeleton_df.columns]
            features_df.columns = [c.strip() for c in features_df.columns]
            if "ctcvr_label" not in skeleton_df.columns:
                skeleton_df["ctcvr_label"] = (
                    skeleton_df["ctr_label"] * skeleton_df["cvr_label"]
                ).astype(np.int8)
        else:
            if data_dir:
                logger.warning("数据文件未找到，使用合成数据: %s", data_dir)
            else:
                logger.info("data_dir 为空，使用合成数据")
            n = sample_size if sample_size else 500_000
            skeleton_df, features_df = generate_synthetic_data(
                n_samples=n, seed=seed
            )
            sample_size = None  # 已经生成了指定数量

        # 采样
        if sample_size and len(skeleton_df) > sample_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(skeleton_df), size=sample_size, replace=False)
            skeleton_df = skeleton_df.iloc[idx].reset_index(drop=True)

        return skeleton_df, features_df

    def _pivot_features(self, features_df, skeleton_df):
        """Long format → wide format，只保留 skeleton 中的 sample_id"""
        valid_ids = set(skeleton_df["sample_id"].tolist())
        features_df = features_df[features_df["sample_id"].isin(valid_ids)]
        wide = features_df.pivot_table(
            index="sample_id", columns="feature_name", values="feature_value", aggfunc="first"
        ).reset_index()
        return wide

    def _fit_transform(self, wide_df, skeleton_df):
        """特征处理并存储为 tensor"""
        # 合并
        merged = skeleton_df.merge(wide_df, on="sample_id", how="left")

        # ── 稀疏特征：整数化 ──────────────────────────────
        self.sparse_vocab = {}   # feat_name → vocab_size
        sparse_arrays = {}
        for feat in SPARSE_FEATURES:
            if feat in merged.columns:
                col = merged[feat].fillna(0).astype(int)
                vocab_size = int(col.max()) + 1
                self.sparse_vocab[feat] = vocab_size
                sparse_arrays[feat] = col.values.astype(np.int64)
            else:
                # 缺失特征用 0 填充，vocab_size=1
                self.sparse_vocab[feat] = 1
                sparse_arrays[feat] = np.zeros(len(merged), dtype=np.int64)

        # ── 数值特征：标准化 ──────────────────────────────
        self.dense_mean = {}
        self.dense_std  = {}
        dense_arrays = {}
        for feat in DENSE_FEATURES:
            if feat in merged.columns:
                col = merged[feat].fillna(0).astype(float).values
                mu, sigma = col.mean(), col.std() + 1e-8
                self.dense_mean[feat] = mu
                self.dense_std[feat]  = sigma
                dense_arrays[feat] = ((col - mu) / sigma).astype(np.float32)
            else:
                self.dense_mean[feat] = 0.0
                self.dense_std[feat]  = 1.0
                dense_arrays[feat] = np.zeros(len(merged), dtype=np.float32)

        # ── 标签 ──────────────────────────────────────────
        self.ctr_labels    = torch.tensor(merged["ctr_label"].values,    dtype=torch.float32)
        self.cvr_labels    = torch.tensor(merged["cvr_label"].values,    dtype=torch.float32)
        self.ctcvr_labels  = torch.tensor(merged["ctcvr_label"].values,  dtype=torch.float32)

        # ── 存储特征 tensor ───────────────────────────────
        self.sparse_tensors = {
            k: torch.tensor(v, dtype=torch.long) for k, v in sparse_arrays.items()
        }
        self.dense_tensor = torch.tensor(
            np.stack([dense_arrays[f] for f in DENSE_FEATURES], axis=1),
            dtype=torch.float32,
        )

        self.n_samples = len(merged)
        logger.info(
            "数据集 [%s] 加载完成: %d 样本 | sparse_vocab=%s",
            self.split, self.n_samples, self.sparse_vocab,
        )

    def _load_from_precomputed(self, d: dict):
        self.sparse_vocab   = d["sparse_vocab"]
        self.dense_mean     = d["dense_mean"]
        self.dense_std      = d["dense_std"]
        self.sparse_tensors = d["sparse_tensors"]
        self.dense_tensor   = d["dense_tensor"]
        self.ctr_labels     = d["ctr_labels"]
        self.cvr_labels     = d["cvr_labels"]
        self.ctcvr_labels   = d["ctcvr_labels"]
        self.n_samples      = d["n_samples"]

    # ── Dataset 接口 ──────────────────────────────────────────

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        features = {k: v[idx] for k, v in self.sparse_tensors.items()}
        features["dense"] = self.dense_tensor[idx]
        return features, self.ctr_labels[idx], self.cvr_labels[idx], self.ctcvr_labels[idx]

    @property
    def feature_info(self) -> Dict:
        """返回特征元信息，供模型初始化使用"""
        return {
            "sparse_vocab":  self.sparse_vocab,    # {feat_name: vocab_size}
            "sparse_feats":  SPARSE_FEATURES,
            "dense_feats":   DENSE_FEATURES,
            "dense_dim":     len(DENSE_FEATURES),
        }


# ─────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────

def build_dataloaders(
    config,
    val_ratio: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    构建 train / val / test DataLoader，并返回 feature_info。

    Parameters
    ----------
    config    : Config
    val_ratio : float  从训练集中划出的验证比例

    Returns
    -------
    train_loader, val_loader, test_loader, feature_info
    """
    train_ds = AliCCPDataset(
        data_dir=config.data_dir,
        split="train",
        sample_size=config.sample_size,
        seed=config.seed,
    )
    test_ds = AliCCPDataset(
        data_dir=config.data_dir,
        split="test",
        sample_size=config.sample_size // 5 if config.sample_size else None,
        seed=config.seed + 1,
    )

    # 从 train 中划出 val
    n_val   = int(len(train_ds) * val_ratio)
    n_train = len(train_ds) - n_val
    train_sub, val_sub = random_split(
        train_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_sub, batch_size=config.batch_size, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader   = DataLoader(
        val_sub,   batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader  = DataLoader(
        test_ds,   batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader, test_loader, train_ds.feature_info
