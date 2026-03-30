#!/usr/bin/env python3
"""
新数据集训练脚本 - 使用 DataLoaders（GPU友好）
数据集: /mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import time


# ===== 数据集类 =====
class IVRDataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        print(f'Loading {data_path}...')
        t0 = time.time()
        
        df = pq.read_table(data_path).to_pandas()
        if max_samples:
            df = df.head(max_samples)
        
        # 特征和标签
        self.labels = torch.FloatTensor(df['ctcvr_label'].values.astype(np.float32))
        self.business_types = df['business_type'].values  # 保持为 numpy 用于分组统计
        
        # 所有特征（126个，全部是类别特征）
        feature_cols = [c for c in df.columns if c not in ['click_label', 'ctcvr_label', 'business_type']]
        self.features = torch.LongTensor(df[feature_cols].values.astype(np.int64))
        
        print(f'Loaded {len(df):,} samples in {time.time()-t0:.1f}s')
        print(f'CVR rate: {self.labels.mean():.4f}')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': self.labels[idx],
            'business_type': self.business_types[idx]
        }


def collate_fn(batch):
    features = torch.stack([b['features'] for b in batch])
    labels = torch.stack([b['labels'] for b in batch])
    business_types = np.array([b['business_type'] for b in batch])
    return {
        'features': features,
        'labels': labels,
        'business_types': business_types
    }


# ===== 模型定义 =====
class BaseModel(nn.Module):
    """改进的 baseline: embedding + MLP + 正确初始化"""
    def __init__(self, vocab_sizes, embed_dim=16, hidden_dim=256):
        super().__init__()
        self.n_features = len(vocab_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # BatchNorm for embeddings
        self.bn = nn.BatchNorm1d(self.n_features * embed_dim)
        
        # MLP with dropout
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Xavier initialization
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, features):
        # features: [batch, n_features]
        embs = []
        for i, emb in enumerate(self.embeddings):
            feat_vals = features[:, i].clamp(0, emb.num_embeddings - 1)  # 防止越界
            embs.append(emb(feat_vals))
        
        x = torch.cat(embs, dim=-1)  # [batch, n_features * embed_dim]
        x = self.bn(x)
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)


# ===== 训练函数 =====
def train_model(model, train_loader, device, epochs=1):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            pred = model(features)
            loss = F.binary_cross_entropy(pred, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if n_batches % 50 == 0:
                print(f'  batch {n_batches}, loss={loss.item():.4f}')
    
    return model


def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_business_types = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['labels']
            business_types = batch['business_types']
            
            pred = model(features)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_business_types.extend(business_types)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_business_types = np.array(all_business_types)
    
    # Overall AUC
    overall_auc = roc_auc_score(all_labels, all_preds)
    
    # Per business_type AUC
    unique_bt = np.unique(all_business_types)
    bt_aucs = {}
    for bt in unique_bt:
        idx = all_business_types == bt
        if idx.sum() > 100 and all_labels[idx].var() > 0:  # 需要有足够的样本且有正负例
            bt_aucs[bt] = roc_auc_score(all_labels[idx], all_preds[idx])
    
    return overall_auc, bt_aucs


# ===== 主流程 =====
def main():
    DATA_DIR = '/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8192
    EPOCHS = 1
    EMBED_DIM = 16
    HIDDEN_DIM = 256
    
    print(f'Device: {DEVICE}')
    print('=' * 60)
    
    # Load vocab sizes
    with open(f'{DATA_DIR}/vocab_sizes.json') as f:
        vocab_sizes = json.load(f)
    
    feature_cols = [c for c in vocab_sizes.keys() if c != 'business_type']
    vocab_sizes_list = [vocab_sizes[c] for c in feature_cols]
    
    print(f'Features: {len(vocab_sizes_list)} (all categorical)')
    print(f'Vocab sizes: min={min(vocab_sizes_list)}, max={max(vocab_sizes_list)}, avg={np.mean(vocab_sizes_list):.1f}')
    
    # Load datasets
    train_ds = IVRDataset(f'{DATA_DIR}/train/')
    test_ds = IVRDataset(f'{DATA_DIR}/test/')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    # Build model
    print(f'\nBuilding model...')
    model = BaseModel(vocab_sizes_list, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train
    print(f'\nTraining...')
    t0 = time.time()
    model = train_model(model, train_loader, DEVICE, EPOCHS)
    print(f'Training done in {time.time()-t0:.1f}s')
    
    # Evaluate
    print(f'\nEvaluating...')
    overall_auc, bt_aucs = evaluate(model, test_loader, DEVICE)
    
    print(f'\nOverall AUC: {overall_auc:.4f}')
    
    # Top 10 business_type by sample count
    test_bt_counts = {}
    for batch in test_loader:
        for bt in batch['business_types']:
            test_bt_counts[bt] = test_bt_counts.get(bt, 0) + 1
    
    top_bt = sorted(test_bt_counts.keys(), key=lambda x: test_bt_counts[x], reverse=True)[:10]
    
    print('\nTop 10 business_type AUC:')
    print(f'{"business_type":<15} {"samples":>10} {"AUC":>8}')
    print('-' * 40)
    for bt in top_bt:
        if bt in bt_aucs:
            print(f'{bt:<15} {test_bt_counts[bt]:>10,} {bt_aucs[bt]:>8.4f}')
        else:
            print(f'{bt:<15} {test_bt_counts[bt]:>10,} {"N/A":>8}')
    
    print('\nDone!')


if __name__ == '__main__':
    main()