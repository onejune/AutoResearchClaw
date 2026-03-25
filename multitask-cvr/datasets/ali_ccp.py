"""
datasets/ali_ccp.py - 真实 Ali-CCP 数据集加载器

数据文件：
  sample_skeleton_{split}.csv   — 曝光日志（4200万行）
  common_features_{split}.csv   — 用户公共特征（73万行）

特征：取 feature_id in KEY_FEATURE_IDS 的 feature_value（每个 field 取第一次出现的值）
      + 13 个统计数值特征（用户/广告/类目维度）
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

# 统计数值特征名称（顺序固定）
DENSE_FEATS = [
    "u_imp_cnt", "u_click_cnt", "u_buy_cnt",
    "u_ctr", "u_cvr", "u_ctcvr",
    "ad_imp_cnt", "ad_click_cnt",
    "ad_ctr", "ad_cvr", "ad_ctcvr",
    "cate_ctr", "cate_ctcvr",
]
DENSE_DIM = len(DENSE_FEATS)   # 13

# feature_id for adgroup_id and cate_id
FID_ADGROUP = 205
FID_CATE    = 206


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


# ─────────────────────────────────────────────────────────────
# 统计特征构造（全量扫描 skeleton）
# ─────────────────────────────────────────────────────────────

def _build_stat_features(path: str) -> Tuple[dict, dict, dict]:
    """
    全量扫描 skeleton，构造用户/广告/类目的统计特征。
    skeleton 格式：sample_id,click,buy,user_id,?,feat_str

    返回：
      user_stats  : {user_id(str): [imp, click, buy]}
      ad_stats    : {adgroup_id(int): [imp, click, buy]}
      cate_stats  : {cate_id(int): [imp, click, buy]}
    """
    logger.info("全量扫描 skeleton 构造统计特征: %s", path)
    user_stats: Dict[str, list] = {}
    ad_stats:   Dict[int, list] = {}
    cate_stats: Dict[int, list] = {}

    need_fids = {FID_ADGROUP, FID_CATE}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            p = line.split(',', 5)
            if len(p) < 4:
                continue
            try:
                click = int(p[1])
                buy   = int(p[2])
            except ValueError:
                continue
            user_id  = p[3]
            feat_str = p[5].rstrip('\n') if len(p) > 5 else ''

            # 用户统计
            if user_id not in user_stats:
                user_stats[user_id] = [0, 0, 0]
            user_stats[user_id][0] += 1
            user_stats[user_id][1] += click
            user_stats[user_id][2] += buy

            # 广告 / 类目统计
            feats = _parse_feature_str(feat_str, need_fids)
            ad_id   = feats.get(FID_ADGROUP)
            cate_id = feats.get(FID_CATE)

            if ad_id is not None:
                if ad_id not in ad_stats:
                    ad_stats[ad_id] = [0, 0, 0]
                ad_stats[ad_id][0] += 1
                ad_stats[ad_id][1] += click
                ad_stats[ad_id][2] += buy

            if cate_id is not None:
                if cate_id not in cate_stats:
                    cate_stats[cate_id] = [0, 0, 0]
                cate_stats[cate_id][0] += 1
                cate_stats[cate_id][1] += click
                cate_stats[cate_id][2] += buy

    logger.info("统计特征扫描完成：%d 用户 | %d 广告 | %d 类目",
                len(user_stats), len(ad_stats), len(cate_stats))
    return user_stats, ad_stats, cate_stats


def _compute_stat_tables(user_stats, ad_stats, cate_stats):
    """
    将原始计数转换为归一化统计特征表。
    log1p + z-score 用于 imp/click/buy 计数；
    比率特征直接计算（已在 [0,1]）。

    返回：
      user_table  : {user_id(str): np.array(6,)}
      ad_table    : {adgroup_id(int): np.array(5,)}
      cate_table  : {cate_id(int): np.array(2,)}
    """

    def _zscore_log1p(counts_list):
        """counts_list: list of int → np.array after log1p z-score"""
        arr = np.log1p(np.array(counts_list, dtype=np.float32))
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-8:
            return arr - mu
        return (arr - mu) / sigma

    # ── 用户 ──────────────────────────────────────────────
    u_ids   = list(user_stats.keys())
    u_imp   = [user_stats[u][0] for u in u_ids]
    u_clk   = [user_stats[u][1] for u in u_ids]
    u_buy   = [user_stats[u][2] for u in u_ids]

    u_imp_z = _zscore_log1p(u_imp)
    u_clk_z = _zscore_log1p(u_clk)
    u_buy_z = _zscore_log1p(u_buy)

    user_table: Dict[str, np.ndarray] = {}
    for i, uid in enumerate(u_ids):
        imp = u_imp[i]; clk = u_clk[i]; buy = u_buy[i]
        u_ctr    = clk / imp if imp > 0 else 0.0
        u_cvr    = buy / clk if clk > 0 else 0.0
        u_ctcvr  = buy / imp if imp > 0 else 0.0
        user_table[uid] = np.array(
            [u_imp_z[i], u_clk_z[i], u_buy_z[i], u_ctr, u_cvr, u_ctcvr],
            dtype=np.float32
        )

    # ── 广告 ──────────────────────────────────────────────
    a_ids   = list(ad_stats.keys())
    a_imp   = [ad_stats[a][0] for a in a_ids]
    a_clk   = [ad_stats[a][1] for a in a_ids]
    a_buy   = [ad_stats[a][2] for a in a_ids]

    a_imp_z = _zscore_log1p(a_imp)
    a_clk_z = _zscore_log1p(a_clk)

    ad_table: Dict[int, np.ndarray] = {}
    for i, aid in enumerate(a_ids):
        imp = a_imp[i]; clk = a_clk[i]; buy = a_buy[i]
        ad_ctr   = clk / imp if imp > 0 else 0.0
        ad_cvr   = buy / clk if clk > 0 else 0.0
        ad_ctcvr = buy / imp if imp > 0 else 0.0
        ad_table[aid] = np.array(
            [a_imp_z[i], a_clk_z[i], ad_ctr, ad_cvr, ad_ctcvr],
            dtype=np.float32
        )

    # ── 类目 ──────────────────────────────────────────────
    cate_table: Dict[int, np.ndarray] = {}
    for cid, (imp, clk, buy) in cate_stats.items():
        cate_ctr   = clk / imp if imp > 0 else 0.0
        cate_ctcvr = buy / imp if imp > 0 else 0.0
        cate_table[cid] = np.array([cate_ctr, cate_ctcvr], dtype=np.float32)

    logger.info("统计特征表构建完成")
    return user_table, ad_table, cate_table


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
                   key_ids: set, user_feat_map: dict,
                   user_table: dict, ad_table: dict, cate_table: dict):
    """
    第二遍扫描 skeleton：解析特征并 join common_features + 统计特征。
    返回 (records, labels, dense_matrix)
    """
    logger.info("加载 skeleton 特征: %s", path)
    records, labels = [], []
    dense_rows = []

    _zero_user = np.zeros(6, dtype=np.float32)
    _zero_ad   = np.zeros(5, dtype=np.float32)
    _zero_cate = np.zeros(2, dtype=np.float32)

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

            # 统计特征
            u_vec    = user_table.get(user_id, _zero_user)
            ad_id    = skel_feats.get(FID_ADGROUP)
            cate_id  = skel_feats.get(FID_CATE)
            ad_vec   = ad_table.get(ad_id,   _zero_ad)   if ad_id   is not None else _zero_ad
            cate_vec = cate_table.get(cate_id, _zero_cate) if cate_id is not None else _zero_cate

            dense_rows.append(np.concatenate([u_vec, ad_vec, cate_vec]))  # (13,)

    logger.info("skeleton 加载完成，共 %d 条记录", len(records))
    dense_matrix = np.stack(dense_rows, axis=0) if dense_rows else np.zeros((0, DENSE_DIM), dtype=np.float32)
    return records, labels, dense_matrix


# ─────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class _AliCCPTorchDataset(Dataset):
    """
    内部 PyTorch Dataset。
    vocab: {fid: {raw_value: local_idx}}  (idx=0 为缺失占位)
    dense_matrix: np.array (N, DENSE_DIM)
    """

    def __init__(self, records, labels, vocab: dict, dense_matrix: np.ndarray):
        n = len(records)
        feat_matrix = np.zeros((n, len(FIELD_LIST)), dtype=np.int64)
        for i, rec in enumerate(records):
            for j, fid in enumerate(FIELD_LIST):
                if fid in rec:
                    feat_matrix[i, j] = vocab[fid].get(rec[fid], 0)

        self.feat_tensor  = torch.tensor(feat_matrix, dtype=torch.long)
        self.dense_tensor = torch.tensor(dense_matrix, dtype=torch.float32)

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
        features["dense"] = self.dense_tensor[idx]
        return features, self.ctr_labels[idx], self.cvr_labels[idx], self.ctcvr_labels[idx]


# ─────────────────────────────────────────────────────────────
# Dataset 加载器
# ─────────────────────────────────────────────────────────────

class AliCCPDataset(BaseMultiTaskDataset):
    """真实 Ali-CCP 数据集加载器"""

    sparse_feats = SPARSE_FEATS
    dense_feats  = DENSE_FEATS
    dense_dim    = DENSE_DIM

    def get_dataloaders(self, config, val_ratio: float = 0.1):
        data_dir = config.data_dir
        if not data_dir:
            raise ValueError("使用 ali_ccp 数据集时，config.data_dir 不能为空")

        sk_path = os.path.join(data_dir, "sample_skeleton_train.csv")

        # ── 全量扫描构造统计特征表 ────────────────────────
        user_stats, ad_stats, cate_stats = _build_stat_features(sk_path)
        user_table, ad_table, cate_table = _compute_stat_tables(
            user_stats, ad_stats, cate_stats
        )

        # ── 第一遍：采样 skeleton，收集涉及的 user_id ────────
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
        records, labels, dense_matrix = _load_skeleton(
            sk_path, sampled, KEY_FEATURE_IDS, user_feat_map,
            user_table, ad_table, cate_table
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
        train_full_ds = _AliCCPTorchDataset(records, labels, vocab, dense_matrix)

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
            # test 集也需要统计特征（复用 train 的统计表）
            test_records, test_labels, test_dense = _load_skeleton(
                test_sk, test_sampled, KEY_FEATURE_IDS, test_user_map,
                user_table, ad_table, cate_table
            )
            test_ds = _AliCCPTorchDataset(test_records, test_labels, vocab, test_dense)
        else:
            logger.warning("test 文件不存在，复用 val 集作为 test")
            test_ds = val_sub

        logger.info(
            "AliCCPDataset 加载完成 | train=%d | val=%d | test=%d | sparse_vocab size=%s | dense_dim=%d",
            n_train, n_val, len(test_ds),
            {k: v for k, v in list(sparse_vocab.items())[:5]},
            DENSE_DIM,
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
            "dense_feats":  DENSE_FEATS,
            "dense_dim":    DENSE_DIM,
        }
        return train_loader, val_loader, test_loader, feature_info
