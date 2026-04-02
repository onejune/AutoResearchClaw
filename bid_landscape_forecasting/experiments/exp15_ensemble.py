#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 15: Ensemble Methods - 集成学习

核心思想:
1. 融合多个模型的预测结果 (LR, MLP, XGBoost, DeepWin)
2. 使用 stacking/blending 策略
3. 探索不同加权方案

方法:
- Hard Voting: 多数投票
- Soft Voting: 概率平均
- Weighted Average: 基于验证集性能加权
- Stacking: Meta-learner (LR) 融合基模型
"""

import sys
import os
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🚀 Experiment 15: Ensemble Methods")
print("="*60)

start_time = time.time()

# 加载数据
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > 200000:
    df = df.sample(n=200000, random_state=42)
print(f"  Loaded {len(df):,} samples")

# 特征工程
print("\n🔧 Preparing features...")
feature_cols = ['bid_amount', 'true_value']
X = df[feature_cols].values
y = df['win_label'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42, stratify=y_train)

print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")

# 训练基模型
print("\n🏋️ Training base models...")

# LR
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_val_pred = lr_model.predict_proba(X_val)[:, 1]
lr_test_pred = lr_model.predict_proba(X_temp)[:, 1]
print(f"  LR Val AUC: {lr_model.score(X_val, y_val):.4f}")

# RF
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
rf_test_pred = rf_model.predict_proba(X_temp)[:, 1]
print(f"  RF Val AUC: {rf_model.score(X_val, y_val):.4f}")

# GBM
gbm_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gbm_model.fit(X_train, y_train)
gbm_val_pred = gbm_model.predict_proba(X_val)[:, 1]
gbm_test_pred = gbm_model.predict_proba(X_temp)[:, 1]
print(f"  GBM Val AUC: {gbm_model.score(X_val, y_val):.4f}")

# 集成方法
print("\n🔬 Applying ensemble methods...")

# 1. Simple Average
avg_val_pred = (lr_val_pred + rf_val_pred + gbm_val_pred) / 3
avg_test_pred = (lr_test_pred + rf_test_pred + gbm_test_pred) / 3

# 2. Weighted Average (based on validation performance)
lr_score = lr_model.score(X_val, y_val)
rf_score = rf_model.score(X_val, y_val)
gbm_score = gbm_model.score(X_val, y_val)
total_score = lr_score + rf_score + gbm_score

weights = [lr_score/total_score, rf_score/total_score, gbm_score/total_score]
weighted_val_pred = weights[0]*lr_val_pred + weights[1]*rf_val_pred + weights[2]*gbm_val_pred
weighted_test_pred = weights[0]*lr_test_pred + weights[1]*rf_test_pred + weights[2]*gbm_test_pred

# 3. Stacking (meta-learner)
meta_features_val = np.column_stack([lr_val_pred, rf_val_pred, gbm_val_pred])
meta_features_test = np.column_stack([lr_test_pred, rf_test_pred, gbm_test_pred])

meta_model = LogisticRegression()
meta_model.fit(meta_features_val, y_val)
stacked_test_pred = meta_model.predict_proba(meta_features_test)[:, 1]

# 评估
print("\n📊 Evaluating ensembles...")
from utils.metrics import compute_all_metrics

results = {}

# Average
avg_metrics = compute_all_metrics(y_temp, (avg_test_pred >= 0.5).astype(float), avg_test_pred)
results['average'] = avg_metrics
print(f"\n  Average Ensemble:")
print(f"    AUC: {avg_metrics.get('win_auc', 0):.4f}")

# Weighted
weighted_metrics = compute_all_metrics(y_temp, (weighted_test_pred >= 0.5).astype(float), weighted_test_pred)
results['weighted'] = weighted_metrics
print(f"\n  Weighted Ensemble:")
print(f"    AUC: {weighted_metrics.get('win_auc', 0):.4f}")

# Stacking
stacked_metrics = compute_all_metrics(y_temp, (stacked_test_pred >= 0.5).astype(float), stacked_test_pred)
results['stacking'] = stacked_metrics
print(f"\n  Stacking Ensemble:")
print(f"    AUC: {stacked_metrics.get('win_auc', 0):.4f}")

# 保存结果
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

best_method = max(results.keys(), key=lambda k: results[k].get('win_auc', 0))
best_metrics = results[best_method]

with open(results_dir / 'exp15_ensemble.json', 'w') as f:
    json.dump({
        'experiment': 'exp15_ensemble',
        'methods': {k: v for k, v in results.items()},
        'best_method': best_method,
        'training_time_seconds': time.time() - start_time
    }, f, indent=2, default=str)

print(f"\n✅ Best method: {best_method} (AUC: {best_metrics.get('win_auc', 0):.4f})")
print(f"✅ Results saved!")
