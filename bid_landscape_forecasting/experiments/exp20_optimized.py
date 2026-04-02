#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 20: 最优特征组合 - 利用 win_prob 和序列特征
目标：接近 DeepWin 的性能
"""

import json, time, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

print("🚀 Exp20: Optimal Feature Combination")
print("=" * 60)

project_root = Path(__file__).parent.parent
start_time = time.time()

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
df = df.sample(n=100000, random_state=42).reset_index(drop=True)
print(f"  Samples: {len(df):,}")

# 特征工程
print("\n📊 Feature engineering...")

# 1. 基础特征
features = ['bid_amount', 'click_label', 'win_prob']

# 2. 序列特征（模拟 DeepWin 的历史竞价）
for seq_len in [3, 5]:
    df[f'bid_ma{seq_len}'] = df['bid_amount'].rolling(seq_len, min_periods=1).mean()
    df[f'bid_std{seq_len}'] = df['bid_amount'].rolling(seq_len, min_periods=1).std().fillna(0)
    features.extend([f'bid_ma{seq_len}', f'bid_std{seq_len}'])

# 3. 交互特征
df['bid_click'] = df['bid_amount'] * df['click_label']
df['bid_sq'] = df['bid_amount'] ** 2
features.extend(['bid_click', 'bid_sq'])

# 4. 分位数特征
df['bid_decile'] = pd.qcut(df['bid_amount'], 10, labels=False, duplicates='drop')
features.append('bid_decile')

print(f"  Features: {features}")
print(f"  Total features: {len(features)}")

# 准备数据
X = df[features].values
y = df['win_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train: {len(X_train):,}, Test: {len(y_test):,}")

# 模型对比
print("\n🏋️ Training models...")

results = {}

# 1. GB
print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
results['GB'] = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])

# 2. RF
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results['RF'] = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

# 3. LR
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
results['LR'] = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

# 4. Simple Neural Network
print("  Training Neural Network...")

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

for epoch in range(50):
    model.train()
    for batch_x, batch_y in train_dl:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    pred_nn = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    results['NN'] = roc_auc_score(y_test, pred_nn)

# 结果
print("\n" + "=" * 60)
print("📊 Results:")
print("-" * 60)
for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<20} AUC = {auc:.4f}")

best_model = max(results, key=results.get)
best_auc = results[best_model]

print(f"\n🏆 Best: {best_model} with AUC = {best_auc:.4f}")
print(f"📊 vs DeepWin: {0.9497 - best_auc:.4f} gap")
print("=" * 60)

# 保存
results_dir = project_root / 'results'
results_dir.mkdir(exist_ok=True)
with open(results_dir / 'exp20_optimized.json', 'w') as f:
    json.dump({
        'experiment': 'exp20_optimized',
        'features': features,
        'results': results,
        'best_auc': best_auc,
        'training_time': time.time() - start_time
    }, f, indent=2)

print("\n✅ Done!")
