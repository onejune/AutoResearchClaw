#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 13 (优化版): DeepHit with Better Architecture

优化点:
1. 使用更强大的网络架构 (ResNet-style)
2. 更好的时间离散化策略
3. 数据增强和正则化
4. 早停和模型选择
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

print("="*60)
print("🚀 Experiment 13 (Optimized): DeepHit")
print("="*60)

start_time = time.time()

# ========== 配置 ==========
config = {
    'max_samples': 200000,
    'n_time_bins': 20,  # 增加时间区间数量
    'hidden_dims': [256, 128, 64],  # 更大的网络
    'dropout': 0.3,
    'batch_size': 512,
    'epochs': 100,
    'lr': 0.0005,
    'patience': 15,
    'random_state': 42
}

# ========== 加载数据 ==========
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
print(f"  Loaded {len(df):,} samples")

if config['max_samples'] and len(df) > config['max_samples']:
    df = df.sample(n=config['max_samples'], random_state=config['random_state'])
    print(f"  Sampled to {len(df):,} samples")

# 特征工程
feature_cols = ['bid_amount', 'true_value', 'click_label']
X = df[feature_cols].fillna(0).values
y = df['win_label'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 时间离散化
times = pd.qcut(df['bid_amount'], q=config['n_time_bins'], labels=False, duplicates='drop').values

# 数据集划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
times_train, times_temp = train_test_split(times, test_size=0.2, random_state=42)

val_ratio = 0.1 / 0.8
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
)
times_train, times_val = train_test_split(times_train, test_size=val_ratio, random_state=42)

print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")
print(f"  Event rate: {y.mean():.4f}")

# ========== DeepHit 模型 (优化版) ==========
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return nn.ReLU()(self.net(x) + self.shortcut(x))

class DeepHitOptimized(nn.Module):
    """优化的 DeepHit 模型"""
    def __init__(self, input_dim, n_time_bins=20, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[i], hidden_dims[i+1]) 
            for i in range(len(hidden_dims)-1)
        ])
        
        # Time-specific heads
        self.time_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(n_time_bins)
        ])
        
        self.n_time_bins = n_time_bins
    
    def forward(self, x):
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        
        # Get risk scores for each time bin
        risks = torch.stack([head(h).squeeze(-1) for head in self.time_heads], dim=1)
        
        # Convert to probabilities using softmax
        probs = nn.Softmax(dim=1)(risks)
        
        return probs

# ========== 损失函数 ==========
def deephit_loss(probs, times, events, alpha=0.5):
    """DeepHit loss: NLL + Ranking"""
    batch_size = probs.size(0)
    
    # NLL loss
    log_probs = torch.log(probs + 1e-10)
    
    # Survival function
    survival = torch.cumprod(1 - probs + 1e-10, dim=1)
    
    # Event loss
    event_mask = (events == 1)
    censor_mask = (events == 0)
    
    if event_mask.sum() > 0:
        nll_event = -log_probs[event_mask, times[event_mask]].mean()
    else:
        nll_event = torch.tensor(0.0).to(probs.device)
    
    if censor_mask.sum() > 0:
        nll_censor = -torch.log(survival[censor_mask, -1] + 1e-10).mean()
    else:
        nll_censor = torch.tensor(0.0).to(probs.device)
    
    nll_loss = (nll_event * event_mask.sum() + nll_censor * censor_mask.sum()) / batch_size
    
    # Ranking loss (simplified)
    ranking_loss = torch.tensor(0.0).to(probs.device)
    if event_mask.sum() > 1:
        event_indices = torch.where(event_mask)[0]
        for i in event_indices:
            t_i = times[i]
            for j in event_indices:
                if i != j and times[j] > t_i:
                    cum_risk_i = probs[i, :t_i+1].sum()
                    cum_risk_j = probs[j, :t_i+1].sum()
                    ranking_loss += torch.relu(cum_risk_j - cum_risk_i + 0.1)
        
        ranking_loss = ranking_loss / (event_mask.sum() ** 2)
    
    return nll_loss + alpha * ranking_loss

# ========== 创建 DataLoader ==========
train_dataset = TensorDataset(
    torch.FloatTensor(X_train),
    torch.LongTensor(times_train),
    torch.FloatTensor(y_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.LongTensor(times_val),
    torch.FloatTensor(y_val)
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

# ========== 模型训练 ==========
print("\n🏗️ Building model...")
model = DeepHitOptimized(
    input_dim=X_scaled.shape[1],
    n_time_bins=config['n_time_bins'],
    hidden_dims=config['hidden_dims']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

print("\n🏋️ Training...")
optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(config['epochs']):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_times, batch_events in train_loader:
        batch_x = batch_x.to(device)
        batch_times = batch_times.to(device)
        batch_events = batch_events.to(device)
        
        optimizer.zero_grad()
        probs = model(batch_x)
        loss = deephit_loss(probs, batch_times, batch_events)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            
            probs = model(batch_x)
            loss = deephit_loss(probs, batch_times, batch_events)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{config['epochs']}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
    
    if patience_counter >= config['patience']:
        print(f"  ⏹️ Early stopping at epoch {epoch+1}")
        break

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print(f"  ✅ Loaded best model (val_loss={best_val_loss:.4f})")

# ========== 评估 ==========
print("\n📈 Evaluating...")
model.eval()

test_X = torch.FloatTensor(X_temp).to(device)
with torch.no_grad():
    probs = model(test_X)
    # Win probability = 1 - survival at last time
    win_prob = (1 - torch.cumprod(1 - probs + 1e-10, dim=1)[:, -1]).cpu().numpy()

auc = roc_auc_score(y_temp, win_prob)
print(f"  🎯 Test AUC: {auc:.4f}")

# ========== 保存结果 ==========
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

results = {
    'experiment': 'exp13_deephit_optimized',
    'paper': 'DeepHit: A Deep Learning Approach to Survival Analysis (AAAI 2018)',
    'config': config,
    'metrics': {
        'auc': auc,
        'best_val_loss': best_val_loss,
        'training_epochs': epoch + 1
    },
    'training_time_seconds': time.time() - start_time,
    'device': str(device),
    'total_params': total_params,
    'improvements': ['residual_blocks', 'larger_network', 'better_discretization', 'early_stopping']
}

with open(results_dir / 'exp13_deephit_optimized.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved! AUC: {auc:.4f}")
print("="*60)
