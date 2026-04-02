#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 17: Calibration Methods - 概率校准

核心思想:
1. 评估不同模型的校准性能 (ECE, MCE, Brier Score)
2. 应用校准方法：Platt Scaling, Isotonic Regression
3. 对比校准前后的性能变化

应用场景:
- 需要可靠概率估计的决策系统
- 风险敏感型任务
- A/B testing 中的 uplift 模型
"""

import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🚀 Experiment 17: Probability Calibration")
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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42, stratify=y_train)

print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")

# ECE 计算
def compute_ece(y_true, y_prob, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece

# 训练未校准模型
print("\n🏋️ Training uncalibrated models...")

# LR
lr_uncal = LogisticRegression(max_iter=1000, random_state=42)
lr_uncal.fit(X_train, y_train)
lr_uncal_pred = lr_uncal.predict_proba(X_temp)[:, 1]
lr_uncal_ece = compute_ece(y_temp, lr_uncal_pred)
print(f"  LR (uncalibrated): ECE = {lr_uncal_ece:.4f}")

# MLP
mlp_uncal = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_uncal.fit(X_train, y_train)
mlp_uncal_pred = mlp_uncal.predict_proba(X_temp)[:, 1]
mlp_uncal_ece = compute_ece(y_temp, mlp_uncal_pred)
print(f"  MLP (uncalibrated): ECE = {mlp_uncal_ece:.4f}")

# 应用校准方法
print("\n🔬 Applying calibration methods...")

# 1. Platt Scaling (Sigmoid)
lr_platt = CalibratedClassifierCV(lr_uncal, method='sigmoid', cv='prefit')
lr_platt.fit(X_val, y_val)
lr_platt_pred = lr_platt.predict_proba(X_temp)[:, 1]
lr_platt_ece = compute_ece(y_temp, lr_platt_pred)
print(f"  LR + Platt Scaling: ECE = {lr_platt_ece:.4f}")

# 2. Isotonic Regression
lr_iso = CalibratedClassifierCV(lr_uncal, method='isotonic', cv='prefit')
lr_iso.fit(X_val, y_val)
lr_iso_pred = lr_iso.predict_proba(X_temp)[:, 1]
lr_iso_ece = compute_ece(y_temp, lr_iso_pred)
print(f"  LR + Isotonic: ECE = {lr_iso_ece:.4f}")

# MLP 校准
mlp_platt = CalibratedClassifierCV(mlp_uncal, method='sigmoid', cv='prefit')
mlp_platt.fit(X_val, y_val)
mlp_platt_pred = mlp_platt.predict_proba(X_temp)[:, 1]
mlp_platt_ece = compute_ece(y_temp, mlp_platt_pred)
print(f"  MLP + Platt Scaling: ECE = {mlp_platt_ece:.4f}")

mlp_iso = CalibratedClassifierCV(mlp_uncal, method='isotonic', cv='prefit')
mlp_iso.fit(X_val, y_val)
mlp_iso_pred = mlp_iso.predict_proba(X_temp)[:, 1]
mlp_iso_ece = compute_ece(y_temp, mlp_iso_pred)
print(f"  MLP + Isotonic: ECE = {mlp_iso_ece:.4f}")

# 结果汇总
print("\n📊 Results Summary:")
results = {
    'lr_uncalibrated': {'ece': lr_uncal_ece},
    'lr_platt': {'ece': lr_platt_ece, 'improvement': lr_uncal_ece - lr_platt_ece},
    'lr_isotonic': {'ece': lr_iso_ece, 'improvement': lr_uncal_ece - lr_iso_ece},
    'mlp_uncalibrated': {'ece': mlp_uncal_ece},
    'mlp_platt': {'ece': mlp_platt_ece, 'improvement': mlp_uncal_ece - mlp_platt_ece},
    'mlp_isotonic': {'ece': mlp_iso_ece, 'improvement': mlp_uncal_ece - mlp_iso_ece}
}

for method, metrics in results.items():
    print(f"  {method}: ECE = {metrics['ece']:.4f}" + 
          (f" (Δ: {metrics.get('improvement', 0):.4f})" if 'improvement' in metrics else ""))

best_method = min({k: v['ece'] for k, v in results.items()}.items(), key=lambda x: x[1])
print(f"\n✅ Best calibration: {best_method[0]} (ECE: {best_method[1]:.4f})")

# 保存结果
print("\n💾 Saving results...")
results_dir = project_root / 'results'
import os
os.makedirs(results_dir, exist_ok=True)

with open(results_dir / 'exp17_calibration.json', 'w') as f:
    json.dump({
        'experiment': 'exp17_calibration',
        'results': results,
        'best_method': best_method[0],
        'training_time_seconds': time.time() - start_time
    }, f, indent=2, default=str)

print("✅ Results saved!")
