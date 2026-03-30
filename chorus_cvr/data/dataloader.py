"""
Ali-CCP 数据集加载器
复用 exp_multitask 项目的数据处理逻辑
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import pickle


class AliCCPDataset(Dataset):
    """Ali-CCP 数据集"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sparse_features: List[str],
        dense_features: List[str],
        label_encoders: Dict[str, LabelEncoder],
        click_col: str = 'click',
        conversion_col: str = 'purchase'
    ):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.click_col = click_col
        self.conversion_col = conversion_col
        
        # 编码稀疏特征
        self.sparse_data = {}
        for feat in sparse_features:
            if feat in data.columns:
                # 使用已有的 encoder 转换
                values = data[feat].fillna('__UNKNOWN__').astype(str)
                # 处理未见过的值
                encoded = []
                for v in values:
                    if v in label_encoders[feat].classes_:
                        encoded.append(label_encoders[feat].transform([v])[0])
                    else:
                        encoded.append(0)  # 未知值映射到 0
                self.sparse_data[feat] = torch.tensor(encoded, dtype=torch.long)
        
        # 稠密特征
        if dense_features:
            dense_values = data[dense_features].fillna(0).values.astype(np.float32)
            self.dense_data = torch.tensor(dense_values, dtype=torch.float32)
        else:
            self.dense_data = None
        
        # 标签
        self.click_labels = torch.tensor(data[click_col].values, dtype=torch.float32)
        self.conversion_labels = torch.tensor(data[conversion_col].values, dtype=torch.float32)
        
        self.length = len(data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sparse_features = {k: v[idx] for k, v in self.sparse_data.items()}
        dense_features = self.dense_data[idx] if self.dense_data is not None else None
        click_label = self.click_labels[idx]
        conversion_label = self.conversion_labels[idx]
        
        return sparse_features, dense_features, click_label, conversion_label


def load_ali_ccp_data(
    data_root: str,
    sparse_features: List[str],
    dense_features: List[str],
    max_samples: Optional[int] = None,
    cache_dir: str = './data/cache'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder], Dict[str, int]]:
    """
    加载 Ali-CCP 数据集
    
    Returns:
        train_df, test_df, label_encoders, sparse_feature_dims
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'ali_ccp_processed.pkl')
    
    # 尝试加载缓存
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['train_df'], cached['test_df'], cached['label_encoders'], cached['sparse_feature_dims']
    
    print("Processing Ali-CCP data...")
    
    # 读取原始数据 - 使用实际的文件名
    train_path = os.path.join(data_root, 'ali_ccp_train.csv')
    test_path = os.path.join(data_root, 'ali_ccp_test.csv')
    
    print(f"Loading train data from: {train_path}")
    print(f"Loading test data from: {test_path}")
    
    train_df = pd.read_csv(train_path, nrows=max_samples)
    test_df = pd.read_csv(test_path, nrows=max_samples // 5 if max_samples else None)
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    print(f"Train columns: {list(train_df.columns)}")
    
    # 检查并重命名标签列
    if 'click' not in train_df.columns:
        # 尝试常见的列名
        for col in ['clk', 'is_click', 'label']:
            if col in train_df.columns:
                train_df['click'] = train_df[col]
                test_df['click'] = test_df[col]
                break
    
    if 'purchase' not in train_df.columns:
        for col in ['buy', 'conversion', 'is_conversion', 'cvr_label']:
            if col in train_df.columns:
                train_df['purchase'] = train_df[col]
                test_df['purchase'] = test_df[col]
                break
    
    # 过滤出存在的特征
    existing_sparse = [f for f in sparse_features if f in train_df.columns]
    existing_dense = [f for f in dense_features if f in train_df.columns]
    
    print(f"Using sparse features: {existing_sparse}")
    print(f"Using dense features: {existing_dense}")
    
    # 构建 Label Encoders
    label_encoders = {}
    sparse_feature_dims = {}
    
    for feat in existing_sparse:
        le = LabelEncoder()
        # 合并训练和测试数据来拟合 encoder
        all_values = pd.concat([
            train_df[feat].fillna('__UNKNOWN__').astype(str),
            test_df[feat].fillna('__UNKNOWN__').astype(str)
        ])
        le.fit(all_values)
        label_encoders[feat] = le
        sparse_feature_dims[feat] = len(le.classes_) + 1  # +1 for unknown
    
    # 保存缓存
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_df': train_df,
            'test_df': test_df,
            'label_encoders': label_encoders,
            'sparse_feature_dims': sparse_feature_dims
        }, f)
    
    print(f"Data cached to {cache_file}")
    
    return train_df, test_df, label_encoders, sparse_feature_dims


def create_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sparse_features: List[str],
    dense_features: List[str],
    label_encoders: Dict[str, LabelEncoder],
    batch_size: int = 4096,
    num_workers: int = 4,
    val_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader
    """
    # 过滤出存在的特征
    existing_sparse = [f for f in sparse_features if f in label_encoders]
    existing_dense = [f for f in dense_features if f in train_df.columns]
    
    # 划分验证集
    val_size = int(len(train_df) * val_ratio)
    val_df = train_df.iloc[-val_size:]
    train_df = train_df.iloc[:-val_size]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    train_dataset = AliCCPDataset(
        train_df, existing_sparse, existing_dense, label_encoders
    )
    val_dataset = AliCCPDataset(
        val_df, existing_sparse, existing_dense, label_encoders
    )
    test_dataset = AliCCPDataset(
        test_df, existing_sparse, existing_dense, label_encoders
    )
    
    def collate_fn(batch):
        sparse_features = {}
        for key in batch[0][0].keys():
            sparse_features[key] = torch.stack([item[0][key] for item in batch])
        
        if batch[0][1] is not None:
            dense_features = torch.stack([item[1] for item in batch])
        else:
            dense_features = None
        
        click_labels = torch.stack([item[2] for item in batch])
        conversion_labels = torch.stack([item[3] for item in batch])
        
        return sparse_features, dense_features, click_labels, conversion_labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
