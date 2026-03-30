#!/usr/bin/env python3
"""
新数据集训练脚本 - 支持CPU/GPU，输出按business_type分组的AUC
数据集: /mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import time

# ===== 配置 =====
DATA_DIR = '/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample'
DEVICE = 'cpu'  # GPU 有数值稳定性问题，强制用 CPU
BATCH_SIZE = 8192
EPOCHS = 1
EMBED_DIM = 8  # 降低维度避免过拟合
HIDDEN_DIM = 128

print(f'Device: {DEVICE}')
print('=' * 60)

# ===== 加载数据 =====
print('Loading data...')
t0 = time.time()

train_df = pq.read_table(f'{DATA_DIR}/train/').to_pandas()
test_df = pq.read_table(f'{DATA_DIR}/test/').to_pandas()

with open(f'{DATA_DIR}/meta.json') as f:
    meta = json.load(f)
with open(f'{DATA_DIR}/vocab_sizes.json') as f:
    vocab_sizes = json.load(f)

feature_cols = meta['feature_cols']
label_col = 'ctcvr_label'

print(f'Train: {len(train_df):,}, Test: {len(test_df):,}')
print(f'Features: {len(feature_cols)}, Label: {label_col}')
print(f'Label rate: train={train_df[label_col].mean():.4f}, test={test_df[label_col].mean():.4f}')
print(f'Loaded in {time.time()-t0:.1f}s')

# ===== 预处理 =====
print('\nPreparing tensors...')

# 训练集
train_features = {}
for col in feature_cols:
    train_features[col] = torch.LongTensor(train_df[col].values.astype(np.int64))
train_labels = torch.FloatTensor(train_df[label_col].values.astype(np.float32))
train_bt = train_df['business_type'].values

# 测试集
test_features = {}
for col in feature_cols:
    test_features[col] = torch.LongTensor(test_df[col].values.astype(np.int64))
test_labels = torch.FloatTensor(test_df[label_col].values.astype(np.float32))
test_bt = test_df['business_type'].values

# ===== 模型定义 =====
class BaseModel(nn.Module):
    """简单 baseline: embedding + MLP"""
    def __init__(self, vocab_sizes, embed_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], embed_dim)
            for col in vocab_sizes
        })
        self.bn = nn.BatchNorm1d(len(vocab_sizes) * embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(len(vocab_sizes) * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, batch):
        embs = []
        for col in self.embeddings:
            idx = batch[col].clamp(0, self.embeddings[col].num_embeddings - 1)
            embs.append(self.embeddings[col](idx))
        x = torch.cat(embs, dim=-1)
        x = self.bn(x)
        return torch.sigmoid(self.mlp(x)).squeeze(-1)

# ===== 训练函数 =====
def train_model(model, train_features, train_labels, device, epochs=1, batch_size=8192):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率
    
    N = len(train_labels)
    n_batches = (N + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, N)
            
            batch = {col: train_features[col][start:end].to(device) for col in train_features}
            y = train_labels[start:end].to(device)
            
            pred = model(batch)
            loss = F.binary_cross_entropy(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if bi % 50 == 0:
                print(f'  batch {bi}/{n_batches}, loss={loss.item():.4f}')
    
    return model

def evaluate(model, test_features, test_labels, test_bt, device, batch_size=8192):
    model.eval()
    preds = []
    
    N = len(test_labels)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = {col: test_features[col][start:end].to(device) for col in test_features}
            pred = model(batch)
            preds.extend(pred.cpu().numpy().tolist())
    
    preds = np.array(preds)
    labels = test_labels.numpy()
    
    # Overall AUC
    overall_auc = roc_auc_score(labels, preds)
    
    # Per business_type AUC
    unique_bt = np.unique(test_bt)
    bt_aucs = {}
    for bt in unique_bt:
        idx = test_bt == bt
        if idx.sum() > 100 and labels[idx].sum() > 0 and labels[idx].sum() < idx.sum():
            bt_aucs[bt] = roc_auc_score(labels[idx], preds[idx])
    
    return overall_auc, bt_aucs, preds

# ===== 主流程 =====
print('\n' + '=' * 60)
print('Training baseline model')
print('=' * 60)

model = BaseModel(vocab_sizes, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

t0 = time.time()
model = train_model(model, train_features, train_labels, DEVICE, EPOCHS, BATCH_SIZE)
print(f'Training done in {time.time()-t0:.1f}s')

print('\nEvaluating...')
overall_auc, bt_aucs, _ = evaluate(model, test_features, test_labels, test_bt, DEVICE, BATCH_SIZE)

print(f'\nOverall AUC: {overall_auc:.4f}')

# Top 10 business_type by sample count
bt_counts = {bt: (test_bt == bt).sum() for bt in bt_aucs}
top_bt = sorted(bt_counts.keys(), key=lambda x: bt_counts[x], reverse=True)[:10]

print('\nTop 10 business_type AUC:')
print(f'{"business_type":<20} {"samples":>10} {"AUC":>8}')
print('-' * 40)
for bt in top_bt:
    print(f'{bt:<20} {bt_counts[bt]:>10,} {bt_aucs[bt]:>8.4f}')

print('\nDone!')
