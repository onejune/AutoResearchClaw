#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 11 (修复版): Quantile Methods for Win Rate Estimation

修复点:
1. 不再预测分位数，而是直接预测 win probability
2. 使用 Gradient Boosting 的概率输出
3. 正确的 AUC 评估
"""

import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🚀 Experiment 11 (Fixed): Quantile-inspired Win Rate Estimation")
print("="*60)

start_time = time.time()

# ========== 配置 ==========
config = {
    'max_samples': 100000,
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'random_state': 42
}

# ========== 加载数据 ==========
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > config['max_samples']:
    df = df.sample(n=config['max_samples'], random_state=config['random_state'])
print(f"  Loaded {len(df):,} samples")

# 准备特征
feature_cols = ['bid_amount', 'click_label']
X = df[feature_cols].fillna(0).values
y = df['win_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
print(f"  Win rate: {y_train.mean():.4f}")

# ========== 训练 ==========
print("\n🏋️ Training Gradient Boosting Classifier...")

model = GradientBoostingClassifier(
    n_estimators=config['n_estimators'],
    max_depth=config['max_depth'],
    learning_rate=config['learning_rate'],
    subsample=config['subsample'],
    random_state=config['random_state'],
    validation_fraction=0.1,
    n_iter_no_change=20,
    verbose=0
)

model.fit(X_train, y_train)

# ========== 评估 ==========
print("\n📈 Evaluating...")

pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pred_proba)
rmse = np.sqrt(mean_squared_error(y_test, pred_proba))

# 计算 ECE
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)

ece = compute_ece(y_test, pred_proba)

# 分位数分析
print("\n📊 Prediction Distribution Analysis:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    pred_q = np.quantile(pred_proba, q)
    actual_rate = y_test[pred_proba <= pred_q].mean() if (pred_proba <= pred_q).sum() > 0 else 0
    print(f"  Q{int(q*100):2d}: pred={pred_q:.4f}, actual_rate={actual_rate:.4f}")

print(f"\n  🎯 AUC: {auc:.4f}")
print(f"  📊 RMSE: {rmse:.4f}")
print(f"  📊 ECE: {ece:.4f}")

# ========== 保存结果 ==========
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

results = {
    'experiment': 'exp11_quantile_simple_fixed',
    'method': 'Gradient Boosting for Win Rate (Quantile-inspired)',
    'version': 'fixed_v2',
    'config': config,
    'metrics': {
        'auc': float(auc),
        'rmse': float(rmse),
        'ece': float(ece)
    },
    'training_time_seconds': time.time() - start_time,
    'n_estimators_used': model.n_estimators_
}

with open(results_dir / 'exp11_quantile_simple_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved! AUC: {auc:.4f}")
print("="*60)
