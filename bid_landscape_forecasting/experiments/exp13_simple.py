#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 13: DeepHit Simplified - 生存分析简化版

简化实现，专注于核心功能验证
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🚀 Experiment 13: DeepHit (Simplified)")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

start_time = time.time()

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
print(f"  Loaded {len(df):,} samples")

# 离散化 bid_amount 为时间区间
print("\n📊 Discretizing time...")
df['time_bin'] = pd.qcut(df['bid_amount'], q=10, labels=False, duplicates='drop')
print(f"  Time bins: {sorted(df['time_bin'].unique())}")

# 准备特征
X = df[['bid_amount', 'true_value']].values
times = df['time_bin'].values
events = df['win_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_temp, times_train, times_temp, events_train, events_temp = train_test_split(
    X_scaled, times, events, test_size=0.2, random_state=42, stratify=events
)
X_train, X_val, times_train, times_val, events_train, events_val = train_test_split(
    X_train, times_train, events_train, test_size=0.125, random_state=42, stratify=events_train
)

print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")
print(f"  Event rate: {events.mean():.4f}")

# 简单神经网络
class SimpleSurvivalNet(nn.Module):
    def __init__(self, input_dim, n_time_bins=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_time_bins)
        )
    
    def forward(self, x):
        return torch.softmax(self.network(x), dim=1)

# 创建 DataLoader
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(times_train),
    torch.FloatTensor(events_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.LongTensor(times_val),
    torch.FloatTensor(events_val)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 创建模型
print("\n🏗️ Building model...")
model = SimpleSurvivalNet(input_dim=2, n_time_bins=10).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# 训练
print("\n🏋️ Training...")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(30):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_times, batch_events in train_loader:
        batch_x = batch_x.to(device)
        batch_times = batch_times.to(device)
        batch_events = batch_events.to(device)
        
        optimizer.zero_grad()
        risk_probs = model(batch_x)
        log_risk_probs = torch.log(risk_probs + 1e-10)
        
        # 简化损失：只考虑观察到的事件
        loss = -torch.mean(log_risk_probs[range(len(batch_times)), batch_times] * batch_events)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_times, batch_events in val_loader:
            batch_x = batch_x.to(device)
            batch_times = batch_times.to(device)
            batch_events = batch_events.to(device)
            
            risk_probs = model(batch_x)
            log_risk_probs = torch.log(risk_probs + 1e-10)
            loss = -torch.mean(log_risk_probs[range(len(batch_times)), batch_times] * batch_events)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/30: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= 5:
        print(f"  ⏹️ Early stopping at epoch {epoch+1}")
        break

# 评估
print("\n📈 Evaluating...")
model.eval()

X_test = torch.FloatTensor(X_temp).to(device)

with torch.no_grad():
    risk_probs = model(X_test)
    # 预测胜率：累积风险
    survival = torch.cumprod(1 - risk_probs, dim=1)
    win_prob = (1 - survival[:, -1]).cpu().numpy()

y_pred = (win_prob >= 0.5).astype(float)
auc = roc_auc_score(events_temp, y_pred)

print(f"\n  AUC: {auc:.4f}")

# 计算其他指标
from utils.metrics import compute_all_metrics
metrics = compute_all_metrics(events_temp, y_pred, win_prob)

# 保存结果
print("\n💾 Saving results...")
results_dir = project_root / 'results'
import os
os.makedirs(results_dir, exist_ok=True)

results_json = {
    'experiment': 'exp13_deephit_simplified',
    'paper': 'DeepHit Simplified',
    'config': {'epochs': 30, 'batch_size': 256, 'lr': 0.001},
    'metrics': metrics,
    'training_time_seconds': time.time() - start_time,
    'device': str(device),
    'total_params': total_params
}

with open(results_dir / 'exp13_deephit.json', 'w') as f:
    json.dump(results_json, f, indent=2, default=str)

md_report = f"""# Experiment 13: DeepHit (Simplified) Results

## Method
- Discrete-time survival analysis
- Neural network for hazard function estimation
- Cumulative risk for win probability prediction

## Configuration
- **Samples**: 100,000
- **Time Bins**: 10
- **Hidden Layers**: 64 → 32
- **Epochs**: 30 (with early stopping)

## Results

### Win Rate Prediction
| Metric | Value |
|--------|-------|
| AUC | {metrics.get('win_auc', auc):.4f} |
| RMSE | {metrics.get('win_rmse', 0):.4f if isinstance(metrics.get('win_rmse'), float) else 'N/A'} |
| MAE | {metrics.get('win_mae', 0):.4f if isinstance(metrics.get('win_mae'), float) else 'N/A'} |
| ECE | {metrics.get('win_ece', 0):.4f if isinstance(metrics.get('win_ece'), float) else 'N/A'} |

### Training Info
- **Device**: {device}
- **Parameters**: {total_params:,}
- **Training Time**: {time.time() - start_time:.2f}s

---
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(results_dir / 'exp13_deephit.md', 'w') as f:
    f.write(md_report)

print(f"  ✅ Results saved!")
print("\n" + "="*60)
print("✅ Experiment 13 completed!")
print("="*60)
