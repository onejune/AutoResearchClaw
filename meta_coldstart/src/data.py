"""
Task 构建与数据加载

每个 campaignset 是一个独立 task：
  - support set：K 条样本（模拟冷启动）
  - query set  ：剩余样本（评估适配效果）
"""
import pickle
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────

class IVRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.features = features
        self.X = torch.tensor(df[features].values, dtype=torch.long)
        self.y = torch.tensor(df["purchase"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────

@dataclass
class Task:
    campaignset_id: int
    business_type:  str
    support_df:     pd.DataFrame   # K-shot
    query_df:       pd.DataFrame   # 评估集
    n_pos_support:  int
    n_pos_query:    int

    def support_loader(self, features, batch_size=256, shuffle=True):
        ds = IVRDataset(self.support_df, features)
        return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)

    def query_loader(self, features, batch_size=2048, shuffle=False):
        ds = IVRDataset(self.query_df, features)
        if len(ds) == 0:
            return None  # 空 query set
        return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)


# ─────────────────────────────────────────────────────────
# TaskBuilder
# ─────────────────────────────────────────────────────────

class TaskBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._load()

    def _load(self):
        logger.info("Loading data...")
        train_df = pd.read_pickle(self.cfg.train_path)
        val_df   = pd.read_pickle(self.cfg.val_path)
        self.all_df = pd.concat([train_df, val_df], ignore_index=True)

        with open(self.cfg.meta_path, "rb") as f:
            meta = pickle.load(f)
        self.features    = meta["features"]
        self.vocab_sizes = meta["vocab_sizes"]

        logger.info(f"Total samples: {len(self.all_df):,}, features: {len(self.features)}")

    def build_tasks(self, k_shot: Optional[int] = None) -> Tuple[List[Task], List[Task]]:
        """构建 meta-train / meta-test task 列表"""
        k = k_shot or self.cfg.k_shot
        cfg = self.cfg

        # 过滤有效 campaignset
        grp = self.all_df.groupby("campaignsetid")
        valid_ids = []
        for csid, df in grp:
            if df["purchase"].sum() >= cfg.min_pos_samples and len(df) >= cfg.min_samples:
                valid_ids.append(csid)

        # 按 campaignsetid 排序，确保可复现
        valid_ids = sorted(valid_ids)
        logger.info(f"Valid campaignsets: {len(valid_ids)}")

        # 划分 meta-train / meta-test
        rng = random.Random(cfg.seed)
        rng.shuffle(valid_ids)
        n_train = int(len(valid_ids) * cfg.meta_train_ratio)
        train_ids = valid_ids[:n_train]
        test_ids  = valid_ids[n_train:]
        logger.info(f"Meta-train tasks: {len(train_ids)}, Meta-test tasks: {len(test_ids)}")

        train_tasks = [self._make_task(csid, k) for csid in train_ids]
        test_tasks  = [self._make_task(csid, k) for csid in test_ids]
        return train_tasks, test_tasks

    def _make_task(self, campaignset_id: int, k_shot: int) -> Task:
        df = self.all_df[self.all_df["campaignsetid"] == campaignset_id].copy()
        btype = df["business_type"].iloc[0]

        # 分层采样 support set（保证正样本）
        pos_df = df[df["purchase"] == 1]
        neg_df = df[df["purchase"] == 0]

        k_pos = min(max(1, k_shot // 10), len(pos_df))   # 至少1个正样本，最多10%
        k_neg = min(k_shot - k_pos, len(neg_df))

        support_pos = pos_df.sample(k_pos,  random_state=self.cfg.seed)
        support_neg = neg_df.sample(k_neg,  random_state=self.cfg.seed)
        support_df  = pd.concat([support_pos, support_neg]).sample(frac=1, random_state=self.cfg.seed)

        support_idx = support_df.index
        query_df    = df.drop(index=support_idx)

        return Task(
            campaignset_id = campaignset_id,
            business_type  = btype,
            support_df     = support_df.reset_index(drop=True),
            query_df       = query_df.reset_index(drop=True),
            n_pos_support  = int(support_df["purchase"].sum()),
            n_pos_query    = int(query_df["purchase"].sum()),
        )

    def global_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """全量数据 loader，用于 global model 训练"""
        bs = batch_size or self.cfg.batch_size
        ds = IVRDataset(self.all_df, self.features)
        return DataLoader(ds, batch_size=bs, shuffle=True)
