"""
Criteo Conversion Logs 数据集加载器

数据格式（19列，tab分隔）：
  col0: click timestamp（整数）
  col1: conversion timestamp（空=无转化=label 0，非空=label 1）
  col2-9: 8个整数特征（连续特征）
  col10-18: 9个类别特征（hash字符串）
"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_dataset/data.txt"


class _CriteoConvTorchDataset(Dataset):
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


class CriteoConvDataset(BaseDataset):
    """
    Criteo Conversion Logs 数据集加载器
    - n_continuous = 8
    - n_categorical = 9
    """

    n_continuous: int = 8
    n_categorical: int = 9

    def get_dataloaders(self, config) -> tuple:
        """返回 (train_loader, test_loader)"""
        data_path = config.data_path if config.data_path else DEFAULT_DATA_PATH
        sample_size = config.sample_size
        cat_vocab_size = config.cat_vocab_size

        logger.info(f"[CriteoConv] 加载数据：{data_path}，采样 {sample_size} 行...")

        labels = []
        cont_feats = []
        cat_feats = []

        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                parts = line.rstrip("\n").split("\t")
                while len(parts) < 19:
                    parts.append("")

                # label：col1 是否有转化
                label = 1 if parts[1].strip() != "" else 0
                labels.append(label)

                # 连续特征：col2-9
                cont = []
                for j in range(2, 2 + self.n_continuous):
                    v = parts[j].strip()
                    try:
                        cont.append(float(v))
                    except ValueError:
                        cont.append(0.0)
                cont_feats.append(cont)

                # 类别特征：col10-18
                cat = []
                for j in range(2 + self.n_continuous, 2 + self.n_continuous + self.n_categorical):
                    v = parts[j].strip()
                    if v == "":
                        cat.append(0)
                    else:
                        cat.append(hash(v) % (cat_vocab_size - 1) + 1)
                cat_feats.append(cat)

        labels_arr = np.array(labels, dtype=np.float32)
        cont_arr = np.array(cont_feats, dtype=np.float32)
        cat_arr = np.array(cat_feats, dtype=np.int64)

        # 预处理连续特征
        cont_arr = np.clip(cont_arr, -1e9, 1e9)
        cont_arr = np.sign(cont_arr) * np.log1p(np.abs(cont_arr))

        # 划分训练/测试集（前 80% 训练，后 20% 测试）
        n = len(labels_arr)
        n_train = int(n * 0.8)

        scaler = StandardScaler()
        cont_train = scaler.fit_transform(cont_arr[:n_train]).astype(np.float32)
        cont_test = scaler.transform(cont_arr[n_train:]).astype(np.float32)

        train_dataset = _CriteoConvTorchDataset(
            cont_train, cat_arr[:n_train], labels_arr[:n_train]
        )
        test_dataset = _CriteoConvTorchDataset(
            cont_test, cat_arr[n_train:], labels_arr[n_train:]
        )

        logger.info(
            f"[CriteoConv] 加载完成：训练 {len(train_dataset):,}，"
            f"测试 {len(test_dataset):,}，正样本率 {labels_arr.mean():.4f}"
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
