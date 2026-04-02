#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 17: LightWin Bid Bucketing

参考论文:
- 《LightWin: Efficient Win Rate Estimation for Large-Scale Industrial RTB Systems》, KDD 2026 (Alibaba)

核心思想:
1. 将 bid_amount 离散化为 N 个区间 (buckets)
2. 每个 bucket 独立子模型，共享底层 embedding
3. 目标：模型大小 < 1MB，推理速度 > 100k QPS
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

print("="*60)
print("🚀 Experiment 17: LightWin Bid Bucketing")
print("="*60)

start_time = time.time()

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
print(f"  Loaded {len(df):,} samples")

# Bid Bucketing
n_buckets = 50
df['bid_bucket'] = pd.cut(df['bid_amount'], bins=n_buckets, labels=False, duplicates='drop')
print(f"  Created {df['bid_bucket'].nunique()} bid buckets")

X = df[['bid_amount', 'true_value']].values
y = df['win_label'].values
buckets = df['bid_bucket'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
bucket_train, bucket_test = train_test_split(buckets, test_size=0.2, random_state=42)

# LightWin Model: Shared bottom + Bucket-specific heads
class LightWinModel(nn.Module):
    def __init__(self, input_dim=2, n_buckets=50, hidden_dim=32):
        super().__init__()
        # Shared bottom (lightweight)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        # Bucket-specific heads
        self.bucket_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_buckets)
        ])
        self.n_buckets = n_buckets
    
    def forward(self, x, buckets):
        shared_rep = self.shared(x)
        outputs = []
        for i, bucket in enumerate(buckets):
            bucket_id = int(bucket.item()) if hasattr(bucket, 'item') else int(bucket)
            if bucket_id < self.n_buckets:
                out = self.bucket_heads[bucket_id](shared_rep[i:i+1])
            else:
                out = self.bucket_heads[-1](shared_rep[i:i+1])
            outputs.append(out)
        return torch.cat(outputs, dim=0).squeeze(-1)

print("\n🏗️ Building LightWin model...")
model = LightWinModel(input_dim=2, n_buckets=n_buckets).to(device)
total_params = sum(p.numel() for p in model.parameters())
model_size_mb = total_params * 4 / 1024 / 1024  # Assume float32
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: {model_size_mb:.2f} MB")

# 训练
print("\n🏋️ Training...")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(20):
    model.train()
    indices = np.random.permutation(len(X_train))
    total_loss = 0
    
    for start_idx in range(0, len(X_train), 256):
        end_idx = min(start_idx + 256, len(X_train))
        batch_indices = indices[start_idx:end_idx]
        
        batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
        batch_y = torch.FloatTensor(y_train[batch_indices]).to(device)
        batch_buckets = torch.LongTensor(bucket_train[batch_indices]).to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X, batch_buckets)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/20: Loss={total_loss/len(X_train)*256:.4f}")

# 评估
print("\n📈 Evaluating...")
model.eval()
test_X = torch.FloatTensor(X_test).to(device)
test_buckets = torch.LongTensor(bucket_test).to(device)

with torch.no_grad():
    outputs = model(test_X, test_buckets)
    pred_probs = torch.sigmoid(outputs).cpu().numpy()

auc = roc_auc_score(y_test, pred_probs)
print(f"  AUC: {auc:.4f}")

# 推理速度测试
print("\n⚡ Speed test...")
n_iterations = 1000
start = time.time()
for _ in range(n_iterations):
    _ = model(test_X, test_buckets)
inference_time = (time.time() - start) / n_iterations * 1000  # ms
qps = 1000 / inference_time
print(f"  Inference time: {inference_time:.2f} ms")
print(f"  QPS: {qps:.0f}")

# 保存结果
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / 'exp17_lightwin.json', 'w') as f:
    json.dump({
        'experiment': 'exp17_lightwin_bucketing',
        'paper': 'LightWin: Efficient Win Rate Estimation (KDD 2026)',
        'config': {'n_buckets': n_buckets},
        'metrics': {
            'auc': auc,
            'model_size_mb': model_size_mb,
            'inference_time_ms': inference_time,
            'qps': qps
        },
        'training_time_seconds': time.time() - start_time,
        'features': ['bid_bucketing', 'lightweight', 'industrial_deployment']
    }, f, indent=2)

print(f"\n✅ Results saved!")
print(f"   AUC: {auc:.4f}, Size: {model_size_mb:.2f}MB, QPS: {qps:.0f}")
print("="*60)
