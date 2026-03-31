"""
数据加载模块
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DataConfig, get_data_path


class IVRDataset(Dataset):
    """IVR 数据集"""
    
    def __init__(
        self, 
        data_path: str,
        vocab_sizes: Dict[str, int],
        feature_cols: List[str],
        label_col: str = 'click_label',
        max_samples: int = None
    ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.vocab_sizes = vocab_sizes
        
        # 找到 business_type 的索引
        self.bt_idx = feature_cols.index('business_type') if 'business_type' in feature_cols else None
        
        print(f"Loading data from {data_path}...")
        
        # 加载数据
        if os.path.isdir(data_path):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_parquet(data_path)
        
        if max_samples:
            df = df.head(max_samples)
        
        # 提取特征和标签
        self.features = df[feature_cols].values.astype(np.int64)
        self.labels = df[label_col].values.astype(np.float32)
        
        # 保存 business_type 用于分组评估
        self.business_types = df['business_type'].values if 'business_type' in df.columns else None
        
        print(f"  Loaded {len(df)} samples")
        print(f"  Label mean: {self.labels.mean():.4f}")
        
        # 打印 business_type 分布
        if self.business_types is not None:
            bt_counts = pd.Series(self.business_types).value_counts()
            print(f"  Business types: {len(bt_counts)} unique")
            print(f"  Top 5 business_type: {dict(bt_counts.head(5))}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


def load_vocab_sizes() -> Dict[str, int]:
    """加载词表大小"""
    vocab_path = get_data_path('vocab')
    with open(vocab_path, 'r') as f:
        vocab_sizes = json.load(f)
    return vocab_sizes


def load_meta() -> Dict:
    """加载元数据"""
    meta_path = get_data_path('meta')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta


def get_feature_cols() -> List[str]:
    """获取特征列名"""
    meta = load_meta()
    return meta['feature_cols']


def create_dataloaders(
    batch_size: int = 1024,
    num_workers: int = 4,
    max_samples: int = None,
    label_col: str = 'click_label'
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """创建数据加载器"""
    vocab_sizes = load_vocab_sizes()
    feature_cols = get_feature_cols()
    
    train_path = get_data_path('train')
    full_dataset = IVRDataset(
        train_path, 
        vocab_sizes, 
        feature_cols,
        label_col=label_col,
        max_samples=max_samples
    )
    
    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(total_size * DataConfig.train_ratio)
    val_size = int(total_size * DataConfig.val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, vocab_sizes


if __name__ == '__main__':
    print("=== 测试数据加载 ===\n")
    train_loader, val_loader, test_loader, vocab_sizes = create_dataloaders(
        batch_size=512,
        num_workers=0,
        max_samples=10000
    )
    
    for features, labels in train_loader:
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        break
