"""
dataset.py
PyTorch Dataset，将 pandas DataFrame 中的字符串特征 hash 为 embedding index。
支持从 parquet 缓存文件加载，避免重复处理。
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


VOCAB_SIZE_DEFAULT = 100_000
NONE_INDEX = 0  # 缺失值 / "none" 映射到 index 0


def load_schema(schema_path: str) -> List[str]:
    """从 combine_schema 文件读取特征列表（每行一个特征名）。"""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"combine_schema 文件不存在：{schema_path}")
    with open(schema_path, "r") as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def hash_feature(value: str, vocab_size: int) -> int:
    """将字符串特征值 hash 为 [1, vocab_size-1] 的整数 index。
    index 0 保留给缺失值/none。
    """
    if not value or value == "none":
        return NONE_INDEX
    return (hash(value) % (vocab_size - 1)) + 1


class IVRDataset(Dataset):
    """
    IVR 广告数据集。

    将 pandas DataFrame 中的字符串特征列 hash 为 embedding index，
    返回 {feature_name: tensor} 字典 + label + domain_indicator。

    Args:
        df: pandas DataFrame，包含特征列、label、domain_indicator
        feature_cols: 特征列名列表
        label_col: 标签列名
        vocab_size: embedding 词表大小（hash 取模）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        vocab_size: int = VOCAB_SIZE_DEFAULT,
    ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.vocab_size = vocab_size

        # 预处理：所有特征列转字符串，缺失填 none
        df = df.copy()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = "none"
            else:
                df[col] = df[col].fillna("none").astype(str)

        # 批量 hash（numpy 向量化）
        self.feature_indices: Dict[str, np.ndarray] = {}
        for col in feature_cols:
            self.feature_indices[col] = np.array(
                [hash_feature(v, vocab_size) for v in df[col].values],
                dtype=np.int64,
            )

        self.labels = df[label_col].values.astype(np.float32)
        self.domains = df["domain_indicator"].values.astype(np.int64) \
            if "domain_indicator" in df.columns else np.zeros(len(df), dtype=np.int64)
        self.length = len(df)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        x = {col: torch.tensor(self.feature_indices[col][idx], dtype=torch.long)
             for col in self.feature_cols}
        x["domain_indicator"] = torch.tensor(self.domains[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, label

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str,
        feature_cols: List[str],
        label_col: str = "label",
        vocab_size: int = VOCAB_SIZE_DEFAULT,
    ) -> "IVRDataset":
        """从 parquet 文件加载数据集。"""
        df = pd.read_parquet(parquet_path)
        return cls(df, feature_cols, label_col, vocab_size)

    @staticmethod
    def save_parquet(df: pd.DataFrame, path: str) -> None:
        """将 DataFrame 保存为 parquet，供后续复用。"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[INFO] 数据已保存：{path}（{len(df)} 行）")
