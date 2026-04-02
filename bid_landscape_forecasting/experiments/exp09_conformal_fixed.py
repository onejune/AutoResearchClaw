#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 09 (修复版): Conformal Prediction for Win Rate Estimation

修复点:
1. 使用 conformal prediction 提供置信区间，同时保留点预测能力
2. 正确的 AUC 评估
3. 更合理的校准集划分
"""

import os, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error

warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("🚀 Experiment 09 (Fixed): Conformal Prediction for Win Rate")
print("="*60)

start_time = time.time()

# ========== 配置 ==========
config = {
    'max_samples': 100000,
    'alpha': 0.1,  # 1 - coverage level
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

# 三分割：train / calibration / test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"  Train: {len(X_train):,}, Calibration: {len(X_cal):,}, Test: {len(X_test):,}")
print(f"  Win rate: {y_train.mean():.4f}")

# ========== 训练基础模型 ==========
print("\n🏋️ Training base model...")

# 使用 Gradient Boosting 作为基础模型
base_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=10
)

base_model.fit(X_train, y_train)

# ========== Conformal Prediction ==========
print("\n📐 Computing conformal intervals...")

# 在校准集上获取预测概率
cal_proba = base_model.predict_proba(X_cal)[:, 1]

# 计算校准分数（nonconformity scores）
# 使用绝对误差作为 nonconformity measure
cal_scores = np.abs(cal_proba - y_cal)

# 计算阈值
n_cal = len(y_cal)
alpha = config['alpha']
threshold = np.quantile(cal_scores, 1 - alpha)

print(f"  Conformity threshold (α={alpha}): {threshold:.4f}")

# ========== 测试集预测 ==========
print("\n📈 Evaluating on test set...")

test_proba = base_model.predict_proba(X_test)[:, 1]

# 构建置信区间
lower_bound = np.maximum(0, test_proba - threshold)
upper_bound = np.minimum(1, test_proba + threshold)

# 点预测的 AUC
auc_point = roc_auc_score(y_test, test_proba)

# 置信区间覆盖率
coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))

# 平均区间宽度
avg_width = np.mean(upper_bound - lower_bound)

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

ece = compute_ece(y_test, test_proba)

print(f"\n  🎯 Point Prediction AUC: {auc_point:.4f}")
print(f"  📊 Coverage Rate: {coverage:.4f} (target: {1-alpha:.4f})")
print(f"  📊 Average Interval Width: {avg_width:.4f}")
print(f"  📊 ECE: {ece:.4f}")

# ========== 保存结果 ==========
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

results = {
    'experiment': 'exp09_conformal_fixed',
    'method': 'Conformal Prediction for Win Rate Estimation',
    'version': 'fixed_v2',
    'config': config,
    'metrics': {
        'auc': float(auc_point),
        'coverage_rate': float(coverage),
        'target_coverage': float(1 - alpha),
        'avg_interval_width': float(avg_width),
        'ece': float(ece)
    },
    'training_time_seconds': time.time() - start_time
}

with open(results_dir / 'exp09_conformal_fixed.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved! AUC: {auc_point:.4f}, Coverage: {coverage:.4f}")
print("="*60)
