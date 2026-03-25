"""
datasets/taobao.py - Taobao UserBehavior 数据集加载器

数据文件：UserBehavior.csv（无表头）
格式：user_id, item_id, category_id, behavior_type, timestamp
behavior_type：pv / cart / fav / buy

样本构造逻辑：
  - 以 pv 记录为基础样本
  - CTR label：该 (user_id, item_id) 是否有 cart 或 buy 行为
  - CVR label：该 (user_id, item_id) 是否有 buy 行为
  - CTCVR label：ctr_label * cvr_label

特征：
  稀疏：user_id, item_id, category_id（hash 截断）
  数值（20维）：
    hour（0-23）, dayofweek（0-6）,
    u_pv_cnt, u_buy_cnt, u_cart_cnt, u_buy_rate, u_cart_rate,
    i_pv_cnt, i_buy_cnt, i_buy_rate, i_cart_rate,
    c_buy_rate,
    uc_buy_rate, uc_cart_rate,          # 用户-类目交叉特征
    i_fav_cnt, i_fav_rate,              # 物品热度特征
    u_fav_cnt, u_active_score,          # 用户活跃度特征
    c_pv_cnt, c_cart_rate               # 类目热度特征
"""
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .base import BaseMultiTaskDataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# vocab size（hash 截断）
# ─────────────────────────────────────────────────────────────
USER_VOCAB_SIZE     = 1_000_000   # 100万
ITEM_VOCAB_SIZE     = 4_000_000   # 400万
CATEGORY_VOCAB_SIZE = 10_000      # 1万

SPARSE_FEATS = ["user_id", "item_id", "category_id"]
DENSE_FEATS  = [
    "hour", "dayofweek",
    "u_pv_cnt", "u_buy_cnt", "u_cart_cnt", "u_buy_rate", "u_cart_rate",
    "i_pv_cnt", "i_buy_cnt", "i_buy_rate", "i_cart_rate",
    "c_buy_rate",
    "uc_buy_rate", "uc_cart_rate",
    "i_fav_cnt", "i_fav_rate",
    "u_fav_cnt", "u_active_score",
    "c_pv_cnt", "c_cart_rate",
]
DENSE_DIM    = 20

SPARSE_VOCAB = {
    "user_id":     USER_VOCAB_SIZE,
    "item_id":     ITEM_VOCAB_SIZE,
    "category_id": CATEGORY_VOCAB_SIZE,
}


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class _TaobaoTorchDataset(Dataset):
    """内部 PyTorch Dataset，持有已构造好的 numpy 数组"""

    def __init__(
        self,
        user_ids:    np.ndarray,   # (N,) int64
        item_ids:    np.ndarray,   # (N,) int64
        cat_ids:     np.ndarray,   # (N,) int64
        dense:       np.ndarray,   # (N, 20) float32
        ctr_labels:  np.ndarray,   # (N,) float32
        cvr_labels:  np.ndarray,   # (N,) float32
        ctcvr_labels: np.ndarray,  # (N,) float32
    ):
        self.user_ids     = torch.tensor(user_ids,     dtype=torch.long)
        self.item_ids     = torch.tensor(item_ids,     dtype=torch.long)
        self.cat_ids      = torch.tensor(cat_ids,      dtype=torch.long)
        self.dense        = torch.tensor(dense,        dtype=torch.float32)
        self.ctr_labels   = torch.tensor(ctr_labels,   dtype=torch.float32)
        self.cvr_labels   = torch.tensor(cvr_labels,   dtype=torch.float32)
        self.ctcvr_labels = torch.tensor(ctcvr_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        features = {
            "user_id":     self.user_ids[idx],
            "item_id":     self.item_ids[idx],
            "category_id": self.cat_ids[idx],
            "dense":       self.dense[idx],
        }
        return (
            features,
            self.ctr_labels[idx],
            self.cvr_labels[idx],
            self.ctcvr_labels[idx],
        )


# ─────────────────────────────────────────────────────────────
# 主数据集类
# ─────────────────────────────────────────────────────────────

class TaobaoDataset(BaseMultiTaskDataset):
    """
    Taobao UserBehavior 数据集加载器。

    Parameters
    ----------
    data_path   : str   UserBehavior.csv 文件路径
    split       : str   "train" / "val" / "test"（内部按比例划分）
    sample_size : int   从 pv 样本中随机采样的条数（默认 200万）
    val_ratio   : float 验证集比例（默认 0.1）
    test_ratio  : float 测试集比例（默认 0.1）
    seed        : int   随机种子
    """

    sparse_feats = SPARSE_FEATS
    dense_feats  = DENSE_FEATS
    dense_dim    = DENSE_DIM

    def __init__(
        self,
        data_path:   str = "",
        split:       str = "train",
        sample_size: int = 2_000_000,
        val_ratio:   float = 0.1,
        test_ratio:  float = 0.1,
        seed:        int = 42,
    ):
        self.data_path   = data_path
        self.split       = split
        self.sample_size = sample_size
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed
        self._dataset    = None   # 懒加载，由 _ensure_loaded() 触发

    def _ensure_loaded(self):
        """确保数据已加载（懒加载）"""
        if self._dataset is not None:
            return
        if not self.data_path:
            raise ValueError("TaobaoDataset: data_path 未设置，请在构造时传入或通过 get_dataloaders 调用")
        arrays = self._load_and_build()
        self._dataset = _TaobaoTorchDataset(*arrays)

    # ── 接口 ──────────────────────────────────────────────

    def __len__(self):
        self._ensure_loaded()
        return len(self._dataset)

    def __getitem__(self, idx):
        self._ensure_loaded()
        return self._dataset[idx]

    @property
    def feature_info(self) -> Dict:
        return {
            "sparse_vocab": SPARSE_VOCAB,
            "sparse_feats": SPARSE_FEATS,
            "dense_feats":  DENSE_FEATS,
            "dense_dim":    DENSE_DIM,
        }

    # ── BaseMultiTaskDataset 接口 ─────────────────────────

    def get_dataloaders(self, config, val_ratio: float = 0.1, temporal: bool = False):
        """
        由工厂调用，返回 (train_loader, val_loader, test_loader, feature_info)。
        此方法内部完成全量加载 + 划分，不依赖 self.split。

        Parameters
        ----------
        temporal : bool
            False（默认）：使用随机切分（_load_taobao，有穿越）
            True：使用时序切分（_load_taobao_temporal，无穿越）
        """
        data_path   = getattr(config, "data_path", "")
        sample_size = getattr(config, "sample_size", 2_000_000) or 2_000_000
        seed        = getattr(config, "seed", 42)
        batch_size  = getattr(config, "batch_size", 4096)
        _val_ratio  = getattr(config, "val_ratio",  val_ratio)
        _test_ratio = getattr(config, "test_ratio", 0.1)
        # config 中的 taobao_temporal 优先级高于参数
        temporal    = getattr(config, "taobao_temporal", temporal)

        if not data_path:
            raise ValueError("使用 taobao 数据集时，config.data_path 不能为空")

        if temporal:
            # 时序切分：_load_taobao_temporal 内部已完成 train/val/test 划分
            logger.info("使用时序切分（无穿越）")
            train_arrays, val_arrays, test_arrays = _load_taobao_temporal(
                data_path=data_path,
                sample_size=sample_size,
                seed=seed,
            )
            ctr = np.concatenate([train_arrays[4], val_arrays[4], test_arrays[4]])
            cvr = np.concatenate([train_arrays[5], val_arrays[5], test_arrays[5]])
            ctcvr = np.concatenate([train_arrays[6], val_arrays[6], test_arrays[6]])

            def _make_ds_from_arrays(arrs):
                return _TaobaoTorchDataset(*arrs)

            train_ds = _make_ds_from_arrays(train_arrays)
            val_ds   = _make_ds_from_arrays(val_arrays)
            test_ds  = _make_ds_from_arrays(test_arrays)
        else:
            # 随机切分（原有逻辑）
            logger.info("使用随机切分（有穿越）")
            arrays = _load_taobao(
                data_path=data_path,
                sample_size=sample_size,
                seed=seed,
            )
            user_ids, item_ids, cat_ids, dense, ctr, cvr, ctcvr = arrays

            n = len(user_ids)
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)

            n_test  = max(1, int(n * _test_ratio))
            n_val   = max(1, int(n * _val_ratio))
            n_train = n - n_val - n_test

            train_idx = idx[:n_train]
            val_idx   = idx[n_train:n_train + n_val]
            test_idx  = idx[n_train + n_val:]

            def _make_ds(sel):
                return _TaobaoTorchDataset(
                    user_ids[sel], item_ids[sel], cat_ids[sel],
                    dense[sel], ctr[sel], cvr[sel], ctcvr[sel],
                )

            train_ds = _make_ds(train_idx)
            val_ds   = _make_ds(val_idx)
            test_ds  = _make_ds(test_idx)

        logger.info(
            "TaobaoDataset 加载完成 | train=%d | val=%d | test=%d",
            len(train_ds), len(val_ds), len(test_ds),
        )
        logger.info(
            "标签分布 | CTR=%.4f | CVR=%.4f | CTCVR=%.4f",
            ctr.mean(), cvr.mean(), ctcvr.mean(),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                                  shuffle=False, num_workers=0)

        feature_info = {
            "sparse_vocab": SPARSE_VOCAB,
            "sparse_feats": SPARSE_FEATS,
            "dense_feats":  DENSE_FEATS,
            "dense_dim":    DENSE_DIM,
        }
        return train_loader, val_loader, test_loader, feature_info

    # ── 内部加载（供 __init__ 使用）─────────────────────

    def _load_and_build(self):
        """加载数据并按 split 划分，返回对应分片的 arrays"""
        arrays_all = _load_taobao(
            data_path=self.data_path,
            sample_size=self.sample_size,
            seed=self.seed,
        )
        user_ids, item_ids, cat_ids, dense, ctr, cvr, ctcvr = arrays_all

        n = len(user_ids)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)

        n_test  = max(1, int(n * self.test_ratio))
        n_val   = max(1, int(n * self.val_ratio))
        n_train = n - n_val - n_test

        if self.split == "train":
            sel = idx[:n_train]
        elif self.split == "val":
            sel = idx[n_train:n_train + n_val]
        else:
            sel = idx[n_train + n_val:]

        return (
            user_ids[sel], item_ids[sel], cat_ids[sel],
            dense[sel], ctr[sel], cvr[sel], ctcvr[sel],
        )


# ─────────────────────────────────────────────────────────────
# 核心加载函数（pandas 版，速度快）
# ─────────────────────────────────────────────────────────────

def _zscore(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std()
    if sigma < 1e-8:
        return np.zeros_like(arr)
    return (arr - mu) / sigma


def _load_taobao(
    data_path:   str,
    sample_size: Optional[int] = 2_000_000,
    seed:        int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    读取 UserBehavior.csv，构造多任务样本（含统计特征）。
    使用 pandas 批量读取，速度远快于逐行解析。

    Returns
    -------
    (user_ids, item_ids, cat_ids, dense, ctr_labels, cvr_labels, ctcvr_labels)
    每个均为 numpy array，长度相同。dense shape: (N, 20)
    """
    logger.info("开始读取 Taobao UserBehavior 数据（pandas）: %s", data_path)

    # ── 用 pandas 读取全量数据 ────────────────────────────
    col_names = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    dtype_map = {
        "user_id": np.int32,
        "item_id": np.int32,
        "category_id": np.int32,
        "behavior_type": str,
        "timestamp": np.int32,
    }

    logger.info("读取 CSV（全量）...")
    df = pd.read_csv(
        data_path,
        header=None,
        names=col_names,
        dtype={"user_id": np.int32, "item_id": np.int32,
               "category_id": np.int32, "timestamp": np.int32},
        engine="c",
    )
    logger.info("CSV 读取完成，总行数: %d", len(df))

    # ── 构造 buy/interact 集合 ────────────────────────────
    buy_df      = df[df["behavior_type"] == "buy"][["user_id", "item_id"]].drop_duplicates()
    interact_df = df[df["behavior_type"].isin(["buy", "cart"])][["user_id", "item_id"]].drop_duplicates()

    buy_set      = set(zip(buy_df["user_id"].values, buy_df["item_id"].values))
    interact_set = set(zip(interact_df["user_id"].values, interact_df["item_id"].values))
    logger.info("buy_set=%d | interact_set=%d", len(buy_set), len(interact_set))

    # ── 计算统计特征 ──────────────────────────────────────
    # 用户统计
    u_pv_s   = df[df["behavior_type"] == "pv"].groupby("user_id").size().rename("u_pv")
    u_buy_s  = df[df["behavior_type"] == "buy"].groupby("user_id").size().rename("u_buy")
    u_cart_s = df[df["behavior_type"] == "cart"].groupby("user_id").size().rename("u_cart")
    u_fav_s  = df[df["behavior_type"] == "fav"].groupby("user_id").size().rename("u_fav")

    u_stats = pd.concat([u_pv_s, u_buy_s, u_cart_s, u_fav_s], axis=1).fillna(0).astype(np.float32)
    u_stats["u_pv_log"]   = np.log1p(u_stats["u_pv"].values)
    u_stats["u_buy_log"]  = np.log1p(u_stats["u_buy"].values)
    u_stats["u_cart_log"] = np.log1p(u_stats["u_cart"].values)

    # z-score 标准化
    for col in ["u_pv_log", "u_buy_log", "u_cart_log"]:
        arr = u_stats[col].values
        mu, sigma = arr.mean(), arr.std()
        u_stats[col] = (arr - mu) / (sigma + 1e-8)

    u_stats["u_buy_rate"]  = u_stats["u_buy"]  / (u_stats["u_pv"] + 1.0)
    u_stats["u_cart_rate"] = u_stats["u_cart"] / (u_stats["u_pv"] + 1.0)

    # 用户活跃度特征
    u_fav_log = np.log1p(u_stats["u_fav"].values)
    mu, sigma = u_fav_log.mean(), u_fav_log.std()
    u_stats["u_fav_cnt"] = (u_fav_log - mu) / (sigma + 1e-8)

    # u_active_score = (buy*3 + cart*2 + fav*1) / (pv+1)，min-max 标准化
    active_raw = (u_stats["u_buy"].values * 3 + u_stats["u_cart"].values * 2 + u_stats["u_fav"].values) / (u_stats["u_pv"].values + 1.0)
    a_min, a_max = active_raw.min(), active_raw.max()
    u_stats["u_active_score"] = (active_raw - a_min) / (a_max - a_min + 1e-8)

    # 物品统计
    i_pv_s   = df[df["behavior_type"] == "pv"].groupby("item_id").size().rename("i_pv")
    i_buy_s  = df[df["behavior_type"] == "buy"].groupby("item_id").size().rename("i_buy")
    i_cart_s = df[df["behavior_type"] == "cart"].groupby("item_id").size().rename("i_cart")
    i_fav_s  = df[df["behavior_type"] == "fav"].groupby("item_id").size().rename("i_fav")

    i_stats = pd.concat([i_pv_s, i_buy_s, i_cart_s, i_fav_s], axis=1).fillna(0).astype(np.float32)
    i_stats["i_pv_log"]  = np.log1p(i_stats["i_pv"].values)
    i_stats["i_buy_log"] = np.log1p(i_stats["i_buy"].values)

    for col in ["i_pv_log", "i_buy_log"]:
        arr = i_stats[col].values
        mu, sigma = arr.mean(), arr.std()
        i_stats[col] = (arr - mu) / (sigma + 1e-8)

    i_stats["i_buy_rate"]  = i_stats["i_buy"]  / (i_stats["i_pv"] + 1.0)
    i_stats["i_cart_rate"] = i_stats["i_cart"] / (i_stats["i_pv"] + 1.0)

    # 物品热度特征
    i_fav_log = np.log1p(i_stats["i_fav"].values)
    mu, sigma = i_fav_log.mean(), i_fav_log.std()
    i_stats["i_fav_cnt"] = (i_fav_log - mu) / (sigma + 1e-8)
    i_stats["i_fav_rate"] = i_stats["i_fav"] / (i_stats["i_pv"] + 1.0)

    # 类目统计
    c_pv_s   = df[df["behavior_type"] == "pv"].groupby("category_id").size().rename("c_pv")
    c_buy_s  = df[df["behavior_type"] == "buy"].groupby("category_id").size().rename("c_buy")
    c_cart_s = df[df["behavior_type"] == "cart"].groupby("category_id").size().rename("c_cart")
    c_stats  = pd.concat([c_pv_s, c_buy_s, c_cart_s], axis=1).fillna(0).astype(np.float32)
    c_stats["c_buy_rate"]  = c_stats["c_buy"]  / (c_stats["c_pv"] + 1.0)
    c_stats["c_cart_rate"] = c_stats["c_cart"] / (c_stats["c_pv"] + 1.0)

    # 类目热度特征
    c_pv_log = np.log1p(c_stats["c_pv"].values)
    mu, sigma = c_pv_log.mean(), c_pv_log.std()
    c_stats["c_pv_cnt"] = (c_pv_log - mu) / (sigma + 1e-8)

    # 用户-类目交叉特征
    logger.info("计算用户-类目交叉特征...")
    uc_pv_s   = df[df["behavior_type"] == "pv"].groupby(["user_id", "category_id"]).size().rename("uc_pv")
    uc_buy_s  = df[df["behavior_type"] == "buy"].groupby(["user_id", "category_id"]).size().rename("uc_buy")
    uc_cart_s = df[df["behavior_type"] == "cart"].groupby(["user_id", "category_id"]).size().rename("uc_cart")
    uc_stats  = pd.concat([uc_pv_s, uc_buy_s, uc_cart_s], axis=1).fillna(0).astype(np.float32)
    uc_stats["uc_buy_rate"]  = uc_stats["uc_buy"]  / (uc_stats["uc_pv"] + 1.0)
    uc_stats["uc_cart_rate"] = uc_stats["uc_cart"] / (uc_stats["uc_pv"] + 1.0)

    logger.info("统计特征计算完成 | 用户数=%d | 物品数=%d | 类目数=%d | 用户-类目对数=%d",
                len(u_stats), len(i_stats), len(c_stats), len(uc_stats))

    # ── 取 pv 样本 ────────────────────────────────────────
    pv_df = df[df["behavior_type"] == "pv"].copy()
    logger.info("pv 样本总数: %d", len(pv_df))

    if sample_size is not None and sample_size < len(pv_df):
        pv_df = pv_df.sample(n=sample_size, random_state=seed)
        logger.info("采样后 pv 样本数: %d", len(pv_df))

    # ── join 统计特征 ─────────────────────────────────────
    pv_df = pv_df.join(
        u_stats[["u_pv_log", "u_buy_log", "u_cart_log", "u_buy_rate", "u_cart_rate",
                 "u_fav_cnt", "u_active_score"]],
        on="user_id", how="left",
    )
    pv_df = pv_df.join(
        i_stats[["i_pv_log", "i_buy_log", "i_buy_rate", "i_cart_rate",
                 "i_fav_cnt", "i_fav_rate"]],
        on="item_id", how="left",
    )
    pv_df = pv_df.join(
        c_stats[["c_buy_rate", "c_cart_rate", "c_pv_cnt"]],
        on="category_id", how="left",
    )

    # 用户-类目交叉特征 merge（MultiIndex 需用 merge）
    uc_merge = uc_stats[["uc_buy_rate", "uc_cart_rate"]].reset_index()
    pv_df = pv_df.merge(uc_merge, on=["user_id", "category_id"], how="left")

    # 填充 NaN（未见过的 id）
    stat_cols = ["u_pv_log", "u_buy_log", "u_cart_log", "u_buy_rate", "u_cart_rate",
                 "u_fav_cnt", "u_active_score",
                 "i_pv_log", "i_buy_log", "i_buy_rate", "i_cart_rate",
                 "i_fav_cnt", "i_fav_rate",
                 "c_buy_rate", "c_cart_rate", "c_pv_cnt",
                 "uc_buy_rate", "uc_cart_rate"]
    pv_df[stat_cols] = pv_df[stat_cols].fillna(0.0)

    # ── 时间特征 ──────────────────────────────────────────
    ts_arr = pv_df["timestamp"].values
    # 向量化计算 hour / dayofweek
    dt_arr = pd.to_datetime(ts_arr, unit="s")
    hour_arr      = dt_arr.hour.values.astype(np.float32)
    dayofweek_arr = dt_arr.dayofweek.values.astype(np.float32)

    # z-score 标准化 hour / dayofweek
    for arr in [hour_arr, dayofweek_arr]:
        mu, sigma = arr.mean(), arr.std()
        arr[:] = (arr - mu) / (sigma + 1e-8)

    # ── 构造 dense 矩阵 ───────────────────────────────────
    n = len(pv_df)
    dense = np.column_stack([
        hour_arr,
        dayofweek_arr,
        pv_df["u_pv_log"].values.astype(np.float32),
        pv_df["u_buy_log"].values.astype(np.float32),
        pv_df["u_cart_log"].values.astype(np.float32),
        pv_df["u_buy_rate"].values.astype(np.float32),
        pv_df["u_cart_rate"].values.astype(np.float32),
        pv_df["i_pv_log"].values.astype(np.float32),
        pv_df["i_buy_log"].values.astype(np.float32),
        pv_df["i_buy_rate"].values.astype(np.float32),
        pv_df["i_cart_rate"].values.astype(np.float32),
        pv_df["c_buy_rate"].values.astype(np.float32),
        pv_df["uc_buy_rate"].values.astype(np.float32),
        pv_df["uc_cart_rate"].values.astype(np.float32),
        pv_df["i_fav_cnt"].values.astype(np.float32),
        pv_df["i_fav_rate"].values.astype(np.float32),
        pv_df["u_fav_cnt"].values.astype(np.float32),
        pv_df["u_active_score"].values.astype(np.float32),
        pv_df["c_pv_cnt"].values.astype(np.float32),
        pv_df["c_cart_rate"].values.astype(np.float32),
    ]).astype(np.float32)

    # ── 稀疏特征（hash 截断）─────────────────────────────
    uid_arr = ((pv_df["user_id"].values.astype(np.int64) % USER_VOCAB_SIZE) + 1)
    iid_arr = ((pv_df["item_id"].values.astype(np.int64) % ITEM_VOCAB_SIZE) + 1)
    cid_arr = ((pv_df["category_id"].values.astype(np.int64) % CATEGORY_VOCAB_SIZE) + 1)

    # ── 标签 ─────────────────────────────────────────────
    uid_raw = pv_df["user_id"].values.astype(np.int32)
    iid_raw = pv_df["item_id"].values.astype(np.int32)

    ctr_labels   = np.array(
        [(int(uid_raw[i]), int(iid_raw[i])) in interact_set for i in range(n)],
        dtype=np.float32,
    )
    cvr_labels   = np.array(
        [(int(uid_raw[i]), int(iid_raw[i])) in buy_set for i in range(n)],
        dtype=np.float32,
    )
    ctcvr_labels = ctr_labels * cvr_labels

    logger.info(
        "样本构造完成 | n=%d | CTR=%.4f | CVR=%.4f | CTCVR=%.4f",
        n, ctr_labels.mean(), cvr_labels.mean(), ctcvr_labels.mean(),
    )

    return uid_arr, iid_arr, cid_arr, dense, ctr_labels, cvr_labels, ctcvr_labels


# ─────────────────────────────────────────────────────────────
# 时序切分版本（无数据穿越）
# ─────────────────────────────────────────────────────────────

def _build_arrays_from_pv(
    pv_df: pd.DataFrame,
    u_stats: pd.DataFrame,
    i_stats: pd.DataFrame,
    c_stats: pd.DataFrame,
    uc_stats: pd.DataFrame,
    buy_set: set,
    interact_set: set,
) -> Tuple[np.ndarray, ...]:
    """
    给定 pv_df 和预计算的统计特征，构造样本 arrays。
    统计特征列名与 _load_taobao 保持完全一致（dense_dim=20）。
    """
    pv_df = pv_df.copy()

    # join 统计特征
    pv_df = pv_df.join(
        u_stats[["u_pv_log", "u_buy_log", "u_cart_log", "u_buy_rate", "u_cart_rate",
                 "u_fav_cnt", "u_active_score"]],
        on="user_id", how="left",
    )
    pv_df = pv_df.join(
        i_stats[["i_pv_log", "i_buy_log", "i_buy_rate", "i_cart_rate",
                 "i_fav_cnt", "i_fav_rate"]],
        on="item_id", how="left",
    )
    pv_df = pv_df.join(
        c_stats[["c_buy_rate", "c_cart_rate", "c_pv_cnt"]],
        on="category_id", how="left",
    )

    uc_merge = uc_stats[["uc_buy_rate", "uc_cart_rate"]].reset_index()
    pv_df = pv_df.merge(uc_merge, on=["user_id", "category_id"], how="left")

    stat_cols = ["u_pv_log", "u_buy_log", "u_cart_log", "u_buy_rate", "u_cart_rate",
                 "u_fav_cnt", "u_active_score",
                 "i_pv_log", "i_buy_log", "i_buy_rate", "i_cart_rate",
                 "i_fav_cnt", "i_fav_rate",
                 "c_buy_rate", "c_cart_rate", "c_pv_cnt",
                 "uc_buy_rate", "uc_cart_rate"]
    pv_df[stat_cols] = pv_df[stat_cols].fillna(0.0)

    # 时间特征
    ts_arr = pv_df["timestamp"].values
    dt_arr = pd.to_datetime(ts_arr, unit="s")
    hour_arr      = dt_arr.hour.values.astype(np.float32)
    dayofweek_arr = dt_arr.dayofweek.values.astype(np.float32)
    for arr in [hour_arr, dayofweek_arr]:
        mu, sigma = arr.mean(), arr.std()
        arr[:] = (arr - mu) / (sigma + 1e-8)

    n = len(pv_df)
    dense = np.column_stack([
        hour_arr,
        dayofweek_arr,
        pv_df["u_pv_log"].values.astype(np.float32),
        pv_df["u_buy_log"].values.astype(np.float32),
        pv_df["u_cart_log"].values.astype(np.float32),
        pv_df["u_buy_rate"].values.astype(np.float32),
        pv_df["u_cart_rate"].values.astype(np.float32),
        pv_df["i_pv_log"].values.astype(np.float32),
        pv_df["i_buy_log"].values.astype(np.float32),
        pv_df["i_buy_rate"].values.astype(np.float32),
        pv_df["i_cart_rate"].values.astype(np.float32),
        pv_df["c_buy_rate"].values.astype(np.float32),
        pv_df["uc_buy_rate"].values.astype(np.float32),
        pv_df["uc_cart_rate"].values.astype(np.float32),
        pv_df["i_fav_cnt"].values.astype(np.float32),
        pv_df["i_fav_rate"].values.astype(np.float32),
        pv_df["u_fav_cnt"].values.astype(np.float32),
        pv_df["u_active_score"].values.astype(np.float32),
        pv_df["c_pv_cnt"].values.astype(np.float32),
        pv_df["c_cart_rate"].values.astype(np.float32),
    ]).astype(np.float32)

    uid_arr = ((pv_df["user_id"].values.astype(np.int64) % USER_VOCAB_SIZE) + 1)
    iid_arr = ((pv_df["item_id"].values.astype(np.int64) % ITEM_VOCAB_SIZE) + 1)
    cid_arr = ((pv_df["category_id"].values.astype(np.int64) % CATEGORY_VOCAB_SIZE) + 1)

    uid_raw = pv_df["user_id"].values.astype(np.int32)
    iid_raw = pv_df["item_id"].values.astype(np.int32)

    ctr_labels   = np.array(
        [(int(uid_raw[i]), int(iid_raw[i])) in interact_set for i in range(n)],
        dtype=np.float32,
    )
    cvr_labels   = np.array(
        [(int(uid_raw[i]), int(iid_raw[i])) in buy_set for i in range(n)],
        dtype=np.float32,
    )
    ctcvr_labels = ctr_labels * cvr_labels

    return uid_arr, iid_arr, cid_arr, dense, ctr_labels, cvr_labels, ctcvr_labels


def _load_taobao_temporal(
    data_path: str,
    sample_size: int = 2_000_000,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    # test_ratio = 1 - train_ratio - val_ratio = 0.15
) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """
    时序切分版本，无数据穿越。

    切分逻辑：
    1. 全量数据按 timestamp 排序
    2. 前 70% 时间段为 history（用于计算统计特征和标签集合）
    3. 后 30% 时间段为 future（用于构造样本）
    4. future 中 pv 记录按时间顺序：前 50% 为 train，中 25% 为 val，后 25% 为 test

    Returns
    -------
    (train_arrays, val_arrays, test_arrays)
    每个 arrays 为 (uid, iid, cid, dense, ctr, cvr, ctcvr) 的 tuple
    """
    logger.info("开始读取 Taobao UserBehavior 数据（时序切分）: %s", data_path)

    col_names = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
    df = pd.read_csv(
        data_path,
        header=None,
        names=col_names,
        dtype={"user_id": np.int32, "item_id": np.int32,
               "category_id": np.int32, "timestamp": np.int32},
        engine="c",
    )
    logger.info("CSV 读取完成，总行数: %d", len(df))

    # ── 按时间戳排序，切分 history / future ──────────────
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_ts = df["timestamp"].quantile(0.70)
    logger.info("时序切分点 timestamp=%.0f（70%% 分位数）", split_ts)

    history_df = df[df["timestamp"] <= split_ts]
    future_df  = df[df["timestamp"] >  split_ts]
    logger.info("history 行数: %d | future 行数: %d", len(history_df), len(future_df))

    # ── 用 history 构造 buy/interact 集合 ────────────────
    buy_df      = history_df[history_df["behavior_type"] == "buy"][["user_id", "item_id"]].drop_duplicates()
    interact_df = history_df[history_df["behavior_type"].isin(["buy", "cart"])][["user_id", "item_id"]].drop_duplicates()
    buy_set      = set(zip(buy_df["user_id"].values, buy_df["item_id"].values))
    interact_set = set(zip(interact_df["user_id"].values, interact_df["item_id"].values))
    logger.info("（history）buy_set=%d | interact_set=%d", len(buy_set), len(interact_set))

    # ── 用 history 计算统计特征 ───────────────────────────
    # 用户统计
    u_pv_s   = history_df[history_df["behavior_type"] == "pv"].groupby("user_id").size().rename("u_pv")
    u_buy_s  = history_df[history_df["behavior_type"] == "buy"].groupby("user_id").size().rename("u_buy")
    u_cart_s = history_df[history_df["behavior_type"] == "cart"].groupby("user_id").size().rename("u_cart")
    u_fav_s  = history_df[history_df["behavior_type"] == "fav"].groupby("user_id").size().rename("u_fav")

    u_stats = pd.concat([u_pv_s, u_buy_s, u_cart_s, u_fav_s], axis=1).fillna(0).astype(np.float32)
    u_stats["u_pv_log"]   = np.log1p(u_stats["u_pv"].values)
    u_stats["u_buy_log"]  = np.log1p(u_stats["u_buy"].values)
    u_stats["u_cart_log"] = np.log1p(u_stats["u_cart"].values)

    for col in ["u_pv_log", "u_buy_log", "u_cart_log"]:
        arr = u_stats[col].values
        mu, sigma = arr.mean(), arr.std()
        u_stats[col] = (arr - mu) / (sigma + 1e-8)

    u_stats["u_buy_rate"]  = u_stats["u_buy"]  / (u_stats["u_pv"] + 1.0)
    u_stats["u_cart_rate"] = u_stats["u_cart"] / (u_stats["u_pv"] + 1.0)

    u_fav_log = np.log1p(u_stats["u_fav"].values)
    mu, sigma = u_fav_log.mean(), u_fav_log.std()
    u_stats["u_fav_cnt"] = (u_fav_log - mu) / (sigma + 1e-8)

    active_raw = (u_stats["u_buy"].values * 3 + u_stats["u_cart"].values * 2 + u_stats["u_fav"].values) / (u_stats["u_pv"].values + 1.0)
    a_min, a_max = active_raw.min(), active_raw.max()
    u_stats["u_active_score"] = (active_raw - a_min) / (a_max - a_min + 1e-8)

    # 物品统计
    i_pv_s   = history_df[history_df["behavior_type"] == "pv"].groupby("item_id").size().rename("i_pv")
    i_buy_s  = history_df[history_df["behavior_type"] == "buy"].groupby("item_id").size().rename("i_buy")
    i_cart_s = history_df[history_df["behavior_type"] == "cart"].groupby("item_id").size().rename("i_cart")
    i_fav_s  = history_df[history_df["behavior_type"] == "fav"].groupby("item_id").size().rename("i_fav")

    i_stats = pd.concat([i_pv_s, i_buy_s, i_cart_s, i_fav_s], axis=1).fillna(0).astype(np.float32)
    i_stats["i_pv_log"]  = np.log1p(i_stats["i_pv"].values)
    i_stats["i_buy_log"] = np.log1p(i_stats["i_buy"].values)

    for col in ["i_pv_log", "i_buy_log"]:
        arr = i_stats[col].values
        mu, sigma = arr.mean(), arr.std()
        i_stats[col] = (arr - mu) / (sigma + 1e-8)

    i_stats["i_buy_rate"]  = i_stats["i_buy"]  / (i_stats["i_pv"] + 1.0)
    i_stats["i_cart_rate"] = i_stats["i_cart"] / (i_stats["i_pv"] + 1.0)

    i_fav_log = np.log1p(i_stats["i_fav"].values)
    mu, sigma = i_fav_log.mean(), i_fav_log.std()
    i_stats["i_fav_cnt"] = (i_fav_log - mu) / (sigma + 1e-8)
    i_stats["i_fav_rate"] = i_stats["i_fav"] / (i_stats["i_pv"] + 1.0)

    # 类目统计
    c_pv_s   = history_df[history_df["behavior_type"] == "pv"].groupby("category_id").size().rename("c_pv")
    c_buy_s  = history_df[history_df["behavior_type"] == "buy"].groupby("category_id").size().rename("c_buy")
    c_cart_s = history_df[history_df["behavior_type"] == "cart"].groupby("category_id").size().rename("c_cart")
    c_stats  = pd.concat([c_pv_s, c_buy_s, c_cart_s], axis=1).fillna(0).astype(np.float32)
    c_stats["c_buy_rate"]  = c_stats["c_buy"]  / (c_stats["c_pv"] + 1.0)
    c_stats["c_cart_rate"] = c_stats["c_cart"] / (c_stats["c_pv"] + 1.0)

    c_pv_log = np.log1p(c_stats["c_pv"].values)
    mu, sigma = c_pv_log.mean(), c_pv_log.std()
    c_stats["c_pv_cnt"] = (c_pv_log - mu) / (sigma + 1e-8)

    # 用户-类目交叉特征
    logger.info("计算用户-类目交叉特征（history）...")
    uc_pv_s   = history_df[history_df["behavior_type"] == "pv"].groupby(["user_id", "category_id"]).size().rename("uc_pv")
    uc_buy_s  = history_df[history_df["behavior_type"] == "buy"].groupby(["user_id", "category_id"]).size().rename("uc_buy")
    uc_cart_s = history_df[history_df["behavior_type"] == "cart"].groupby(["user_id", "category_id"]).size().rename("uc_cart")
    uc_stats  = pd.concat([uc_pv_s, uc_buy_s, uc_cart_s], axis=1).fillna(0).astype(np.float32)
    uc_stats["uc_buy_rate"]  = uc_stats["uc_buy"]  / (uc_stats["uc_pv"] + 1.0)
    uc_stats["uc_cart_rate"] = uc_stats["uc_cart"] / (uc_stats["uc_pv"] + 1.0)

    logger.info("统计特征计算完成（history）| 用户数=%d | 物品数=%d | 类目数=%d | 用户-类目对数=%d",
                len(u_stats), len(i_stats), len(c_stats), len(uc_stats))

    # ── 取 future 中的 pv 记录 ───────────────────────────
    future_pv = future_df[future_df["behavior_type"] == "pv"].copy()
    future_pv = future_pv.sort_values("timestamp").reset_index(drop=True)
    logger.info("future pv 样本总数: %d", len(future_pv))

    if sample_size is not None and sample_size < len(future_pv):
        rng = np.random.default_rng(seed)
        sampled_idx = rng.choice(len(future_pv), size=sample_size, replace=False)
        sampled_idx.sort()  # 保持时序顺序
        future_pv = future_pv.iloc[sampled_idx].reset_index(drop=True)
        logger.info("采样后 future pv 样本数: %d", len(future_pv))

    # ── future pv 按时间顺序切分 train/val/test ──────────
    n_future = len(future_pv)
    n_train  = int(n_future * 0.50)
    n_val    = int(n_future * 0.25)
    # n_test = n_future - n_train - n_val

    train_pv = future_pv.iloc[:n_train]
    val_pv   = future_pv.iloc[n_train:n_train + n_val]
    test_pv  = future_pv.iloc[n_train + n_val:]
    logger.info("future 切分 | train=%d | val=%d | test=%d", len(train_pv), len(val_pv), len(test_pv))

    # ── 构造各分片的 arrays ───────────────────────────────
    train_arrays = _build_arrays_from_pv(train_pv, u_stats, i_stats, c_stats, uc_stats, buy_set, interact_set)
    val_arrays   = _build_arrays_from_pv(val_pv,   u_stats, i_stats, c_stats, uc_stats, buy_set, interact_set)
    test_arrays  = _build_arrays_from_pv(test_pv,  u_stats, i_stats, c_stats, uc_stats, buy_set, interact_set)

    logger.info(
        "时序切分完成 | train CTR=%.4f CVR=%.4f | val CTR=%.4f CVR=%.4f | test CTR=%.4f CVR=%.4f",
        train_arrays[4].mean(), train_arrays[5].mean(),
        val_arrays[4].mean(),   val_arrays[5].mean(),
        test_arrays[4].mean(),  test_arrays[5].mean(),
    )

    return train_arrays, val_arrays, test_arrays
