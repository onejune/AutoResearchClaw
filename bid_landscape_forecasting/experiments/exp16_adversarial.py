#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 16: Adversarial Co-Training

参考论文:
- 《Neural Bid Landscape Learning with Counterfactual Imputation》, ICLR 2026 (Oral)

核心思想:
1. Generator: 生成反事实市场价格
2. Discriminator: 区分真实/生成的样本 + 预测胜率
3. 联合训练：L = L_supervised + λ·L_adversarial
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
print("🚀 Experiment 16: Adversarial Co-Training")
print("="*60)

start_time = time.time()

# 简化实现：Generator + Discriminator
class Generator(nn.Module):
    def __init__(self, input_dim=2, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(x.size(0), 16).to(x.device)
        price = self.net(torch.cat([x, z], dim=1))
        return price.squeeze(-1)  # Return 1D tensor

class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 64),  # +1 for generated price
            nn.ReLU(),
            nn.Linear(64, 2)  # real/fake + win/lose
        )
    
    def forward(self, x, price):
        return self.net(torch.cat([x, price.unsqueeze(1)], dim=1))

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)
print(f"  Loaded {len(df):,} samples")

X = df[['bid_amount', 'true_value']].values
y = df['win_label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 训练
print("\n🏋️ Training adversarial model...")
G = Generator().to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=0.001)
opt_D = optim.Adam(D.parameters(), lr=0.001)

criterion_ce = nn.CrossEntropyLoss()

for epoch in range(30):
    for _ in range(5):  # Mini-batch loop
        idx = np.random.choice(len(X_train), 256, replace=False)
        batch_X = torch.FloatTensor(X_train[idx]).to(device)
        batch_y = torch.LongTensor(y_train[idx]).to(device)
        
        # Train Discriminator
        opt_D.zero_grad()
        real_price = torch.FloatTensor(df['true_value'].values[idx]).to(device)
        fake_price = G(batch_X)
        
        real_labels = torch.zeros(256).long().to(device)
        fake_labels = torch.ones(256).long().to(device)
        
        d_real = D(batch_X, real_price)
        d_fake = D(batch_X, fake_price)
        
        loss_D = criterion_ce(d_real, real_labels) + criterion_ce(d_fake, fake_labels)
        loss_D.backward()
        opt_D.step()
        
        # Train Generator
        opt_G.zero_grad()
        d_fake = D(batch_X, G(batch_X))
        loss_G = criterion_ce(d_fake, real_labels)  # Try to fool discriminator
        loss_G.backward()
        opt_G.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/30: Loss_G={loss_G.item():.4f}, Loss_D={loss_D.item():.4f}")

# 评估
print("\n📈 Evaluating...")
G.eval()
test_X = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    pred_price = G(test_X)
    # Use generated price to predict win rate
    pred_win = (pred_price > 0.5).float()

auc = roc_auc_score(y_test, pred_win.cpu().numpy())
print(f"  AUC: {auc:.4f}")

# 保存结果
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / 'exp16_adversarial.json', 'w') as f:
    json.dump({
        'experiment': 'exp16_adversarial_cotraining',
        'paper': 'Neural Bid Landscape Learning with Counterfactual Imputation (ICLR 2026)',
        'metrics': {'auc': auc},
        'training_time_seconds': time.time() - start_time,
        'features': ['generator', 'discriminator', 'adversarial_training']
    }, f, indent=2)

print(f"\n✅ Results saved! AUC: {auc:.4f}")
print("="*60)
