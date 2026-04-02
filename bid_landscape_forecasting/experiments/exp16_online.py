#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 16: Online Learning - 增量学习

核心思想:
1. 模拟在线学习环境，数据流式到达
2. 使用 Passive-Aggressive / Perceptron 等在线算法
3. 评估模型在概念漂移下的鲁棒性

应用场景:
- RTB 环境中的实时模型更新
- 处理数据分布变化
- 降低重新训练成本
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🚀 Experiment 16: Online Learning")
print("="*60)

start_time = time.time()

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
print(f"  Loaded {len(df):,} samples")

# 特征工程
feature_cols = ['bid_amount', 'true_value']
X = df[feature_cols].values
y = df['win_label'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集 (模拟时间序列)
train_size = int(len(X_scaled) * 0.7)
val_size = int(len(X_scaled) * 0.15)

X_train = X_scaled[:train_size]
y_train = y[:train_size]
X_val = X_scaled[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X_scaled[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

# 在线学习
print("\n🏋️ Training online models...")

# 1. Passive-Aggressive (use decision function for probability)
pa_model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
pa_model.fit(X_train, y_train)
pa_decision = pa_model.decision_function(X_test)
pa_test_pred = 1 / (1 + np.exp(-pa_decision))  # sigmoid
pa_auc = roc_auc_score(y_test, pa_test_pred)
print(f"  Passive-Aggressive AUC: {pa_auc:.4f}")

# 2. SGD (Logistic)
sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
sgd_model.fit(X_train, y_train)
sgd_test_pred = sgd_model.predict_proba(X_test)[:, 1]
sgd_auc = roc_auc_score(y_test, sgd_test_pred)
print(f"  SGD Logistic AUC: {sgd_auc:.4f}")

# 3. Partial Fit (增量学习)
print("\n🔄 Testing partial fit (incremental learning)...")
incremental_model = SGDClassifier(loss='log_loss', max_iter=1, random_state=42)

batch_size = 1000
n_batches = len(X_train) // batch_size
auc_history = []

for i in range(n_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    
    incremental_model.partial_fit(
        X_train[start_idx:end_idx], 
        y_train[start_idx:end_idx],
        classes=[0, 1]
    )
    
    if (i + 1) % 100 == 0:
        temp_pred = incremental_model.predict_proba(X_val)[:, 1]
        temp_auc = roc_auc_score(y_val, temp_pred)
        auc_history.append(temp_auc)
        print(f"  Batch {i+1}/{n_batches}: Val AUC = {temp_auc:.4f}")

# Final evaluation
incremental_test_pred = incremental_model.predict_proba(X_test)[:, 1]
incremental_auc = roc_auc_score(y_test, incremental_test_pred)
print(f"\n  Incremental Learning Final AUC: {incremental_auc:.4f}")

# 结果汇总
print("\n📊 Results Summary:")
results = {
    'passive_aggressive': {'auc': pa_auc},
    'sgd_logistic': {'auc': sgd_auc},
    'incremental_learning': {'auc': incremental_auc, 'auc_history': auc_history[-10:]}
}

for method, metrics in results.items():
    print(f"  {method}: AUC = {metrics['auc']:.4f}")

best_method = max(results.keys(), key=lambda k: results[k]['auc'])
print(f"\n✅ Best method: {best_method} (AUC: {results[best_method]['auc']:.4f})")

# 保存结果
print("\n💾 Saving results...")
results_dir = project_root / 'results'
import os
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / 'exp16_online_learning.json', 'w') as f:
    json.dump({
        'experiment': 'exp16_online_learning',
        'results': results,
        'best_method': best_method,
        'training_time_seconds': time.time() - start_time
    }, f, indent=2, default=str)

print("✅ Results saved!")
