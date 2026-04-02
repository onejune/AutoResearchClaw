#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 18: Online Incremental Learning

参考论文:
- 《LightWin: Efficient Win Rate Estimation for Large-Scale Industrial RTB Systems》, KDD 2026 (Alibaba)

核心思想:
1. 模拟流式数据场景
2. 每小时增量更新一次模型
3. 评估：稳定性 (CPM 波动) + 适应性 (概念漂移)
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

print("="*60)
print("🚀 Experiment 18: Online Incremental Learning")
print("="*60)

start_time = time.time()

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
print(f"  Loaded {len(df):,} samples")

X = df[['bid_amount', 'true_value']].values
y = df['win_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 模拟时间序列：按顺序划分成多个 "小时"
n_hours = 10
hourly_size = len(X_scaled) // n_hours
hours_data = []
for i in range(n_hours):
    start_idx = i * hourly_size
    end_idx = (i + 1) * hourly_size if i < n_hours - 1 else len(X_scaled)
    hours_data.append({
        'X': X_scaled[start_idx:end_idx],
        'y': y[start_idx:end_idx]
    })

print(f"  Simulated {n_hours} hours of data")

# 简单 MLP 模型
class OnlineMLP(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = OnlineMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 在线学习：增量更新
print("\n🔄 Online learning with incremental updates...")
auc_history = []
loss_history = []

for hour in range(n_hours):
    hour_data = hours_data[hour]
    X_hour = torch.FloatTensor(hour_data['X']).to(device)
    y_hour = torch.FloatTensor(hour_data['y']).to(device)
    
    # Incremental update (partial fit)
    model.train()
    optimizer.zero_grad()
    outputs = model(X_hour)
    loss = criterion(outputs, y_hour)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    # Evaluate on current data
    model.eval()
    with torch.no_grad():
        pred = model(X_hour).cpu().numpy()
    
    if len(np.unique(hour_data['y'])) > 1:
        auc = roc_auc_score(hour_data['y'], pred)
    else:
        auc = 0.5
    auc_history.append(auc)
    
    if (hour + 1) % 3 == 0 or hour == 0:
        print(f"  Hour {hour+1}/{n_hours}: AUC={auc:.4f}, Loss={loss.item():.4f}")

# 稳定性分析
auc_std = np.std(auc_history)
auc_mean = np.mean(auc_history)
auc_drift = auc_history[-1] - auc_history[0]

print(f"\n📊 Stability Analysis:")
print(f"  Mean AUC: {auc_mean:.4f}")
print(f"  AUC Std: {auc_std:.4f} (lower is more stable)")
print(f"  AUC Drift: {auc_drift:.4f} (adaptation to concept drift)")

# 保存结果
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / 'exp18_online_learning.json', 'w') as f:
    json.dump({
        'experiment': 'exp18_online_incremental',
        'paper': 'LightWin: Efficient Win Rate Estimation (KDD 2026)',
        'config': {'n_hours': n_hours},
        'metrics': {
            'mean_auc': auc_mean,
            'auc_stability': auc_std,
            'auc_drift': auc_drift,
            'final_auc': auc_history[-1],
            'auc_history': auc_history
        },
        'training_time_seconds': time.time() - start_time,
        'features': ['online_learning', 'incremental_update', 'concept_drift']
    }, f, indent=2)

print(f"\n✅ Results saved!")
print(f"   Final AUC: {auc_history[-1]:.4f}, Stability: {auc_std:.4f}, Drift: {auc_drift:.4f}")
print("="*60)
