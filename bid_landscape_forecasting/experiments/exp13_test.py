#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""exp13 快速测试版"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("="*60)
print("🚀 Exp13 Test - DeepHit")
print("="*60)

project_root = Path(__file__).parent.parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
print(f"  Loaded {len(df):,} samples")

# 离散化时间
print("\n📊 Discretizing time...")
df['time_bin'] = pd.qcut(df['bid_amount'], q=10, labels=False, duplicates='drop')
print(f"  Time bins: {sorted(df['time_bin'].unique())}")

# 准备特征
print("\n🔧 Preparing features...")
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

print("\n✅ Data preparation complete!")
print("\nTest successful!")
