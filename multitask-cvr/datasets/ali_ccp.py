"""
datasets/ali_ccp.py - 真实 Ali-CCP 数据集加载器

数据文件：
  sample_skeleton_{split}.csv   — 曝光日志（4200万行）
  common_features_{split}.csv   — 用户公共特征（73万行）

特征：取 feature_id in KEY_FEATURE_IDS 的 feature_value（每个 field 取第一次出现的值）
标签：ctr=click, cvr=buy, ctcvr=click*buy
"""
import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Optional, Tuple

from .base import BaseMultiTaskDataset

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 关键 feature_id 配置
# ─────────────────────────────────────────────────────────────

KEY_FEATURE_IDS = {101, 121, 122, 124, 125, 126, 127, 128, 129,
                   205, 206, 207, 210, 216, 301}

# 有序列表（保证每次字段顺序一致）
FIELD_LIST = sorted(KEY_FEATURE_IDS)

# feature_id → 字段名
FIELD_NAMES = {fid: f"f_{fid}" for fid in FIELD_LIST}

SPARSE_FEATS = [FIELD_NAMES[fid] for fid in FIELD_LIST]


# ─────────────────────────────────────────────────────────────
# 解析工具
# ─────────────────────────────────────────────────────────────

def _parse_feature_str(feat_str: str, key_ids: set) -> dict:
    """
    解析特征串，返回 {feature_id(int): feature_value(int)} 字典。
    格式：feature_id\x02feature_value\x03weight[\x01...]
    只保留 key_ids 中的纯数字 feature_id，每个 field 取第一次出现的值。
    """
    result = {}
    if not feat_str or not isinstance(feat_str, str):
        return result
    for feat in feat_str.split('\x01'):
        parts = feat.split('\x02')
        if len(parts) < 2:
            continue
        fid_str = parts[0].strip()
        if not fid_str.isdigit():
            continue
        fid = int(fid_str)
        if fid not in key_ids or fid in result:
            continue
        val_str = parts[1].split('\x03')[0].strip()
        try:
            result[fid] = int(val_str)
        except ValueError:
            try:
                result[fid] = int(float(val_str))
            except ValueError:
                pass
    return result


def _load_common_features(path: str, key_ids: set,
                           filter_users: Optional[set] = None) -> dict:
    """
    加载 common_features CSV → {user_id: {fid: value}}
    格式：user_id,特征数,特征串

    filter_users: 若非 None，只加载其中的 user_id（大幅加速）
    """
    logger.info("加载 common_features: %s%s", path,
                f"（过滤 {len(filter_users)} 个用户）" if filter_users else "")
    user_feat_map: dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            idx1 = line.find(',')
            if idx1 < 0:
                continue
            user_id = line[:idx1]
            if filter_users is not None and user_id not in filter_users:
                continue
            idx2 = line.find(',', idx1 + 1)
            if idx2 < 0:
                continue
            feat_str = line[idx2 + 1:]
            feats = _parse_feature_str(feat_str, key_ids)
            if user_id not in user_feat_map:
                user_feat_map[user_id] = feats
            else:
                for k, v in feats.items():
                    user_feat_map[user_id].setdefault(k, v)
    logger.info("common_features 加载完成，共 %d 个用户", len(user_feat_map))
    return user_feat_map


def _sample_skeleton_lines(path: str, sample_size: Optional[int], seed: int):
    """
    第一遍扫描 skeleton：决定采样行号，同时收集对应的 user_id 集合。
    返回 (sampled_set_or_None, user_ids_set)
    """
    logger.info("扫描 skeleton 收集 user_id: %s", path)
    if sample_size is not None:
        # 统计总行数
        total = 0
        with open(path, 'r', encoding='utf-8') as f:
            for _ in f:
                total += 1
        logger.info("总行数: %d，采样: %d", total, sample_size)
        rng = np.random.default_rng(seed)
        sampled = set(rng.choice(total, size=min(sample_size, total),
                                 replace=False).tolist())
    else:
        sampled = None

    user_ids_set: set = set()
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if sampled is not None and idx not in sampled:
                continue
            # 只需要 user_id（第4列）
            p = line.split(',', 4)
            if len(p) >= 4:
                user_ids_set.add(p[3])

    logger.info("采样行涉及 %d 个唯一 user_id", len(user_ids_set))
    return sampled, user_ids_set


def _stratified_sample_skeleton_lines(path: str,
                                       neg_sample_rate: float,
                                       click_sample_rate: float,
                                       seed: int):
    """
    分层采样扫描 skeleton（第一遍）：
    - click=0：以 neg_sample_rate 概率保留
    - click=1, buy=0：以 click_sample_rate 概率保留
    - buy=1：全部保留
    返回 (sampled_set, user_ids_set)
    """
    logger.info("分层采样扫描 skeleton: %s (neg_rate=%.3f, click_rate=%.3f)",
                path, neg_sample_rate, click_sample_rate)
    rng = np.random.default_rng(seed)
    sampled: set = set()
    user_ids_set: set = set()

    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            p = line.split(',', 5)
            if len(p) < 4:
                continue
            try:
                click = int(p[1])
                buy   = int(p[2])
            except ValueError:
                continue

            if buy == 1:
                keep = True
            elif click == 1:
                keep = rng.random() < click_sample_rate
            else:
                keep = rng.random() < neg_sample_rate

            if keep:
                sampled.add(idx)
                user_ids_set.add(p[3])

    logger.info("分层采样完成：保留 %d 行，涉及 %d 个唯一 user_id",
                len(sampled), len(user_ids_set))
    return sampled, user_ids_set


def _load_skeleton(path: str, sampled: Optional[set],
                   key_ids: set, user_feat_map: dict):
    """
    第二遍扫描 skeleton：解析特征并 join common_features。
    返回 (records, labels)
    """
    logger.info("加载 skeleton 特征: %s", path)
    records, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if sampled is not None and idx not in sampled:
                continue
            line = line.rstrip('\n')
            parts = line.split(',', 5)
            if len(parts) < 4:
                continue
            try:
                click = int(parts[1])
                buy   = int(parts[2])
            except ValueError:
                continue
            user_id  = parts[3]
            feat_str = parts[5] if len(parts) > 5 else ''

            skel_feats = _parse_feature_str(feat_str, key_ids)
            merged: dict = {}
            if user_id in user_feat_map:
                merged.update(user_feat_map[user_id])
            merged.update(skel_feats)   # skeleton 优先

            records.append(merged)
            labels.append((click, buy, click * buy))

    logger.info("skeleton 加载完成，共 %d 条记录", len(records))
    return records, labels


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class _AliCCPTorchDataset(Dataset):
    """
    内部 PyTorch Dataset。
    vocab: {fid: {raw_value: local_idx}}  (idx=0 为缺失占位)
    """

    def __init__(self, records, labels, vocab: dict):
        n = len(records)
        feat_matrix = np.zeros((n, len(FIELD_LIST)), dtype=np.int64)
        for i, rec in enumerate(records):
            for j, fid in enumerate(FIELD_LIST):
                if fid in rec:
                    feat_matrix[i, j] = vocab[fid].get(rec[fid], 0)

        self.feat_tensor  = torch.tensor(feat_matrix, dtype=torch.long)
        labels_arr        = np.array(labels, dtype=np.float32)
        self.ctr_labels   = torch.tensor(labels_arr[:, 0], dtype=torch.float32)
        self.cvr_labels   = torch.tensor(labels_arr[:, 1], dtype=torch.float32)
        self.ctcvr_labels = torch.tensor(labels_arr[:, 2], dtype=torch.float32)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        features = {
            FIELD_NAMES[fid]: self.feat_tensor[idx, j]
            for j, fid in enumerate(FIELD_LIST)
        }
        return features, self.ctr_labels[idx], self.cvr_labels[idx], self.ctcvr_labels[idx]


# ─────────────────────────────────────────────────────────────
# Dataset 加载器
# ─────────────────────────────────────────────────────────────

class AliCCPDataset(BaseMultiTaskDataset):
    """真实 Ali-CCP 数据集加载器"""

    sparse_feats = SPARSE_FEATS
    dense_feats  = []
    dense_dim    = 0

    def get_dataloaders(self, config, val_ratio: float = 0.1):
        data_dir = config.data_dir
        if not data_dir:
            raise ValueError("使用 ali_ccp 数据集时，config.data_dir 不能为空")

        # ── 第一遍：采样 skeleton，收集涉及的 user_id ────────
        sk_path = os.path.join(data_dir, "sample_skeleton_train.csv")

        neg_rate   = getattr(config, 'ali_ccp_neg_sample_rate', None)
        click_rate = getattr(config, 'ali_ccp_click_sample_rate', None)
        use_stratified = (neg_rate is not None and click_rate is not None
                          and neg_rate > 0 and click_rate > 0)

        if use_stratified:
            sampled, user_ids_set = _stratified_sample_skeleton_lines(
                sk_path, neg_rate, click_rate, config.seed
            )
        else:
            sampled, user_ids_set = _sample_skeleton_lines(
                sk_path, config.sample_size, config.seed
            )

        # ── 只加载采样行涉及的 common_features ───────────
        cf_path = os.path.join(data_dir, "common_features_train.csv")
        user_feat_map = _load_common_features(cf_path, KEY_FEATURE_IDS,
                                              filter_users=user_ids_set)

        # ── 第二遍：加载 skeleton 特征并 join ─────────────
        records, labels = _load_skeleton(
            sk_path, sampled, KEY_FEATURE_IDS, user_feat_map
        )

        if not records:
            raise ValueError(f"skeleton 数据为空: {sk_path}")

        # ── 构建 vocab（在 train 数据上）─────────────────
        vocab: Dict[int, Dict[int, int]] = {}
        for fid in FIELD_LIST:
            val_set = set()
            for rec in records:
                if fid in rec:
                    val_set.add(rec[fid])
            vocab[fid] = {v: i + 1 for i, v in enumerate(sorted(val_set))}

        sparse_vocab = {
            FIELD_NAMES[fid]: len(vocab[fid]) + 1
            for fid in FIELD_LIST
        }

        # ── 构建 train PyTorch Dataset ────────────────────
        train_full_ds = _AliCCPTorchDataset(records, labels, vocab)

        n_val   = int(len(train_full_ds) * val_ratio)
        n_train = len(train_full_ds) - n_val
        train_sub, val_sub = random_split(
            train_full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )

        # ── 加载 test ─────────────────────────────────────
        test_sk = os.path.join(data_dir, "sample_skeleton_test.csv")
        test_cf = os.path.join(data_dir, "common_features_test.csv")
        if os.path.exists(test_sk) and os.path.exists(test_cf):
            if use_stratified:
                test_sampled, test_uids = _stratified_sample_skeleton_lines(
                    test_sk, neg_rate, click_rate, config.seed + 1
                )
            else:
                test_sample = (config.sample_size // 5) if config.sample_size else None
                test_sampled, test_uids = _sample_skeleton_lines(
                    test_sk, test_sample, config.seed + 1
                )
            test_user_map = _load_common_features(test_cf, KEY_FEATURE_IDS,
                                                  filter_users=test_uids)
            test_records, test_labels = _load_skeleton(
                test_sk, test_sampled, KEY_FEATURE_IDS, test_user_map
            )
            test_ds = _AliCCPTorchDataset(test_records, test_labels, vocab)
        else:
            logger.warning("test 文件不存在，复用 val 集作为 test")
            test_ds = val_sub

        logger.info(
            "AliCCPDataset 加载完成 | train=%d | val=%d | test=%d | sparse_vocab size=%s",
            n_train, n_val, len(test_ds),
            {k: v for k, v in list(sparse_vocab.items())[:5]},
        )

        train_loader = DataLoader(train_sub, batch_size=config.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_sub,   batch_size=config.batch_size,
                                  shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,   batch_size=config.batch_size,
                                  shuffle=False, num_workers=0)

        feature_info = {
            "sparse_vocab": sparse_vocab,
            "sparse_feats": SPARSE_FEATS,
            "dense_feats":  [],
            "dense_dim":    0,
        }
        return train_loader, val_loader, test_loader, feature_info
