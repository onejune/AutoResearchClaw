"""
Criteo Standard Display Advertising 数据集加载器

数据格式（parquet）：
  label: int32（0/1）
  I1-I13: float64（13个连续特征，含大量 NaN）
  C1-C26: int32/float64（26个类别特征，hash 值，可能为负数）

规模：~4125 万行
正样本率：~25.6%
"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = (
    "/mnt/data/oss_wanjun/pai_work/open_research/dataset/"
    "criteo_standard/criteo-parquet/train_train.parquet"
)

CONT_COLS = [f"I{i}" for i in range(1, 14)]   # I1-I13
CAT_COLS  = [f"C{i}" for i in range(1, 27)]   # C1-C26


class _CriteoStdTorchDataset(Dataset):
    """内部 PyTorch Dataset"""

    def __init__(self, cont_arr, cat_arr, labels):
        self.cont_arr = cont_arr
        self.cat_arr = cat_arr
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.cont_arr[idx], dtype=torch.float32),
            torch.tensor(self.cat_arr[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class CriteoStdDataset(BaseDataset):
    """
    Criteo Standard Display Advertising 数据集加载器
    - n_continuous = 13
    - n_categorical = 26
    """

    n_continuous: int = 13
    n_categorical: int = 26

    def get_dataloaders(self, config) -> tuple:
        """返回 (train_loader, test_loader)"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装 pandas：pip install pandas pyarrow")

        data_path = config.data_path if config.data_path else DEFAULT_DATA_PATH
        sample_size = config.sample_size
        cat_vocab_size = config.cat_vocab_size
        seed = config.seed

        logger.info(f"[CriteoStd] 加载数据：{data_path}，采样 {sample_size} 行...")

        # 读取 parquet
        df = pd.read_parquet(data_path, columns=["label"] + CONT_COLS + CAT_COLS)

        # 随机采样（sample_size=0 表示全量）
        if sample_size > 0 and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        logger.info(f"[CriteoStd] 采样后行数：{len(df):,}，正样本率：{df['label'].mean():.4f}")

        # ── 连续特征处理 ──
        # NaN 填 0，log1p 变换，StandardScaler 标准化
        cont_arr = df[CONT_COLS].fillna(0.0).values.astype(np.float32)
        cont_arr = np.clip(cont_arr, -1e9, 1e9)
        cont_arr = np.sign(cont_arr) * np.log1p(np.abs(cont_arr))

        # ── 类别特征处理 ──
        # int32 hash 值（可能为负数），取绝对值后 mod vocab_size
        cat_arr = df[CAT_COLS].fillna(0).values.astype(np.int64)
        cat_arr = np.abs(cat_arr) % cat_vocab_size

        # ── 标签 ──
        labels_arr = df["label"].values.astype(np.float32)

        # ── 划分训练/测试集（前 80% 训练，后 20% 测试）──
        n = len(labels_arr)
        n_train = int(n * 0.8)

        scaler = StandardScaler()
        cont_train = scaler.fit_transform(cont_arr[:n_train]).astype(np.float32)
        cont_test = scaler.transform(cont_arr[n_train:]).astype(np.float32)

        train_dataset = _CriteoStdTorchDataset(
            cont_train, cat_arr[:n_train], labels_arr[:n_train]
        )
        test_dataset = _CriteoStdTorchDataset(
            cont_test, cat_arr[n_train:], labels_arr[n_train:]
        )

        logger.info(
            f"[CriteoStd] 加载完成：训练 {len(train_dataset):,}，"
            f"测试 {len(test_dataset):,}"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, test_loader
