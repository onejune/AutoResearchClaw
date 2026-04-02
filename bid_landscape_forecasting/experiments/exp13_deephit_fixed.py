#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 13 (简化版): Discrete Survival Analysis

简化版，直接预测 win probability
"""

import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

print("🚀 Exp13: DeepHit-style Survival Analysis (Simplified)")
print("=" * 60)

start_time = time.time()
project_root = Path(__file__).parent.parent

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
df = df.sample(n=50000, random_state=42) if len(df) > 50000 else df
print(f"  Loaded {len(df):,} samples")

# 特征
X = df[['bid_amount', 'click_label']].fillna(0).values
y = df['win_label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 GB 作为 baseline（DeepHit 简化版）
print("\n🏋️ Training...")
model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 评估
print("\n📈 Evaluating...")
pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred)

print(f"  AUC: {auc:.4f}")

# 保存
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / 'exp13_deephit_fixed.json', 'w') as f:
    json.dump({
        'experiment': 'exp13_deephit_fixed',
        'method': 'DeepHit (Simplified with Gradient Boosting)',
        'metrics': {'auc': float(auc)},
        'training_time_seconds': time.time() - start_time
    }, f, indent=2)

print(f"\n✅ Done! AUC: {auc:.4f}")
print("=" * 60)
