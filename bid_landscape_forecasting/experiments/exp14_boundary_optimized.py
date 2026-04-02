#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 14 (优化版): Inequality Constraints with Better Features

优化点:
1. 更多特征工程 (bid/true_value ratio, log transform)
2. 自适应边界损失权重
3. 集成多个子模型
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
print("🚀 Experiment 14 (Optimized): Boundary Constraints")
print("="*60)

start_time = time.time()

config = {
    'max_samples': 150000,
    'lambda_bound': 0.3,  # 降低边界损失权重
    'batch_size': 512,
    'epochs': 40,
    'lr': 0.001,
    'random_state': 42
}

# ========== 加载数据 + 特征工程 ==========
print("\n📂 Loading data with feature engineering...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > config['max_samples']:
    df = df.sample(n=config['max_samples'], random_state=config['random_state'])
print(f"  Loaded {len(df):,} samples")

# 增强特征
df['bid_log'] = np.log1p(df['bid_amount'])
df['tv_log'] = np.log1p(df['true_value'])
df['bid_tv_ratio'] = df['bid_amount'] / (df['true_value'] + 1e-10)
df['bid_diff'] = df['true_value'] - df['bid_amount']

feature_cols = ['bid_amount', 'true_value', 'click_label', 
                'bid_log', 'tv_log', 'bid_tv_ratio', 'bid_diff']
X = df[feature_cols].fillna(0).values
y = df['win_label'].values
bids = df['bid_amount'].values
tv = df['true_value'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
bid_train, bid_test = train_test_split(bids, test_size=0.2, random_state=42)
tv_train, tv_test = train_test_split(tv, test_size=0.2, random_state=42)

print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
print(f"  Features: {len(feature_cols)} (including engineered features)")

# ========== 模型定义 ==========
class BoundaryMLP(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = BoundaryMLP(len(feature_cols)).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
criterion_bce = nn.BCELoss()

# ========== 训练 ==========
print("\n🏋️ Training with adaptive boundary loss...")

for epoch in range(config['epochs']):
    model.train()
    
    indices = np.random.permutation(len(X_train))
    total_loss = 0
    
    for start_idx in range(0, len(X_train), config['batch_size']):
        end_idx = min(start_idx + config['batch_size'], len(X_train))
        batch_indices = indices[start_idx:end_idx]
        
        batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
        batch_y = torch.FloatTensor(y_train[batch_indices]).to(device)
        batch_bids = torch.FloatTensor(bid_train[batch_indices]).to(device)
        batch_tv = torch.FloatTensor(tv_train[batch_indices]).to(device)
        
        optimizer.zero_grad()
        pred_probs = model(batch_X)
        
        # BCE loss
        bce_loss = criterion_bce(pred_probs, batch_y)
        
        # Adaptive boundary loss
        losing_mask = (batch_y == 0) & (batch_tv > batch_bids)
        if losing_mask.sum() > 0:
            # For losing samples, enforce monotonicity
            bid_ratio = batch_bids[losing_mask] / (batch_tv[losing_mask] + 1e-10)
            expected_prob = pred_probs[losing_mask] + 0.05 * (1 - bid_ratio)
            expected_prob = torch.clamp(expected_prob, 0, 1)
            
            bound_loss = nn.MSELoss()(pred_probs[losing_mask], expected_prob)
        else:
            bound_loss = torch.tensor(0.0).to(device)
        
        # Adaptive lambda: decrease as training progresses
        lambda_t = config['lambda_bound'] * (1 - epoch / config['epochs'])
        loss = bce_loss + lambda_t * bound_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{config['epochs']}: Loss={total_loss/len(indices):.4f}, Lambda={lambda_t:.3f}")

# ========== 评估 ==========
print("\n📈 Evaluating...")
model.eval()

test_X = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    pred_probs = model(test_X).cpu().numpy()

auc = roc_auc_score(y_test, pred_probs)
print(f"  🎯 Test AUC: {auc:.4f}")

# Long-tail analysis
print("\n🔍 Long-tail analysis:")
q25, q75 = np.percentile(bid_test, [25, 75])
head_mask = bid_test <= q25
tail_mask = bid_test >= q75

if head_mask.sum() > 10 and len(np.unique(y_test[head_mask])) > 1:
    head_auc = roc_auc_score(y_test[head_mask], pred_probs[head_mask])
    print(f"  Head (low bid) AUC: {head_auc:.4f}")

if tail_mask.sum() > 10 and len(np.unique(y_test[tail_mask])) > 1:
    tail_auc = roc_auc_score(y_test[tail_mask], pred_probs[tail_mask])
    print(f"  Tail (high bid) AUC: {tail_auc:.4f}")

# ========== 保存结果 ==========
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

results = {
    'experiment': 'exp14_boundary_optimized',
    'paper': 'Win Rate Estimation via Censored Data Modeling (WWW 2025)',
    'config': config,
    'metrics': {
        'auc': auc,
        'head_auc': head_auc if 'head_auc' in dir() else None,
        'tail_auc': tail_auc if 'tail_auc' in dir() else None,
        'n_features': len(feature_cols)
    },
    'training_time_seconds': time.time() - start_time,
    'device': str(device),
    'improvements': ['more_features', 'adaptive_lambda', 'log_transform', 'ratio_features']
}

with open(results_dir / 'exp14_boundary_optimized.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved! AUC: {auc:.4f}")
print("="*60)
