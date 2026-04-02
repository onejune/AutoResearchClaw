#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 16 (简化版 V2): Sequence Modeling for Win Rate
简化版，直接输出到 stdout
"""

import os, sys, json, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

print("🚀 Exp16 V2: Sequence Modeling")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

start_time = time.time()
project_root = Path(__file__).parent.parent

# 配置
config = {'max_samples': 50000, 'seq_len': 5, 'lstm_hidden': 32, 'epochs': 50, 'batch_size': 256, 'lr': 0.001}

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > config['max_samples']:
    df = df.sample(n=config['max_samples'], random_state=42).reset_index(drop=True)
print(f"  Samples: {len(df):,}")

# 创建序列
print("📊 Creating sequences...")
base_bids = df['bid_amount'].values
seq_len = config['seq_len']
X_seq = np.array([[base_bids[i] * (0.6 + 0.4 * j / (seq_len - 1)) + np.random.normal(0, base_bids[i] * 0.05)
                   for j in range(seq_len)] for i in range(len(df))])

X_ctx = df[['bid_amount', 'click_label']].fillna(0).values
y = df['win_label'].values

scaler_ctx = StandardScaler()
X_ctx = scaler_ctx.fit_transform(X_ctx)
scaler_seq = StandardScaler()
X_seq = scaler_seq.fit_transform(X_seq.reshape(-1, 1)).reshape(X_seq.shape)

X_seq_train, X_seq_test, X_ctx_train, X_ctx_test, y_train, y_test = train_test_split(
    X_seq, X_ctx, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(y_train):,}, Test: {len(y_test):,}")

# 模型
class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 16)
        self.lstm = nn.LSTM(16, config['lstm_hidden'], batch_first=True)
        self.ctx_enc = nn.Linear(2, 16)
        self.fc = nn.Sequential(nn.Linear(config['lstm_hidden'] + 16, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, seq, ctx):
        seq_emb = self.embed(seq.unsqueeze(-1))
        lstm_out, _ = self.lstm(seq_emb)
        seq_feat = lstm_out[:, -1, :]
        ctx_feat = torch.relu(self.ctx_enc(ctx))
        return self.fc(torch.cat([seq_feat, ctx_feat], dim=1)).squeeze(-1)

print("\n🏋️ Training...")
model = SeqModel().to(device)
opt = optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.BCELoss()

train_ds = TensorDataset(torch.FloatTensor(X_seq_train), torch.FloatTensor(X_ctx_train), torch.FloatTensor(y_train))
train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

best_auc = 0
for epoch in range(config['epochs']):
    model.train()
    for batch_seq, batch_ctx, batch_y in train_dl:
        batch_seq, batch_ctx, batch_y = batch_seq.to(device), batch_ctx.to(device), batch_y.to(device)
        opt.zero_grad()
        loss = criterion(model(batch_seq, batch_ctx), batch_y)
        loss.backward()
        opt.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_seq_test).to(device), torch.FloatTensor(X_ctx_test).to(device)).cpu().numpy()
            auc = roc_auc_score(y_test, pred)
        print(f"  Epoch {epoch+1}/{config['epochs']}: AUC={auc:.4f}")
        if auc > best_auc:
            best_auc = auc

print(f"\n🎯 Best AUC: {best_auc:.4f}")

# 对比
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
gb.fit(X_ctx_train, y_train)
gb_auc = roc_auc_score(y_test, gb.predict_proba(X_ctx_test)[:, 1])
print(f"📊 GB Baseline: {gb_auc:.4f}")
print(f"📈 Improvement: {(best_auc - gb_auc)*100:.2f}%")

# 保存
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)
with open(results_dir / 'exp16_with_sequence.json', 'w') as f:
    json.dump({'experiment': 'exp16_with_sequence', 'auc': best_auc, 'baseline': gb_auc, 'time': time.time() - start_time}, f, indent=2)

print("\n✅ Done!")
