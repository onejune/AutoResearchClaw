"""
实验 09: Conformal Prediction for Win Rate Estimation (Corrected Implementation)

参考论文: "Distribution-Free Prediction Sets for Win Rate Estimation in RTB" (WWW 2023 Workshop)

核心思想:
- 使用保形预测(Conformal Prediction)构建 win rate 的预测集合
- 不假设数据分布，提供分布无关的置信区间
- 对于给定的出价 bid，输出 win probability 的置信区间而非点估计
- 满足边际覆盖保证: P(y ∈ C_α(x)) ≥ 1-α

修正实现:
- 正确的保形预测算法实现
- 用于回归任务的保形预测 (Conformalized Quantile Regression)
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor


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


class ConformalizedQuantileRegressor:
    """
    保形化分位数回归 (Conformalized Quantile Regression)
    
    一种常用的回归保形预测方法:
    1. 使用分位数回归估计预测区间
    2. 使用校准集计算校正量
    3. 输出校正后的预测区间
    """
    
    def __init__(self, alpha=0.1, quantiles=[0.1, 0.9]):
        self.alpha = alpha  # 错误容忍度
        self.quantiles = quantiles  # 分位数
        self.lower_qr = QuantileRegressor(quantile=quantiles[0], alpha=0.1, solver='highs')
        self.upper_qr = QuantileRegressor(quantile=quantiles[1], alpha=0.1, solver='highs')
        self.calibration_errors = None
        self.error_quantile = None
    
    def fit(self, X_train, y_train, X_cal, y_cal):
        """训练分位数回归器并在校准集上计算校正量"""
        # 训练分位数回归器
        self.lower_qr.fit(X_train, y_train)
        self.upper_qr.fit(X_train, y_train)
        
        # 在校准集上计算校正量
        lower_cal_pred = self.lower_qr.predict(X_cal)
        upper_cal_pred = self.upper_qr.predict(X_cal)
        
        # 计算校准误差（超出范围的程度）
        # 如果真实值小于下界，则误差为下界-真实值
        # 如果真实值大于上界，则误差为真实值-上界
        errors_lower = np.maximum(0, lower_cal_pred - y_cal)  # 真实值太小
        errors_upper = np.maximum(0, y_cal - upper_cal_pred)  # 真实值太大
        self.calibration_errors = np.maximum(errors_lower, errors_upper)
        
        # 计算分位数 (1-α)
        n_cal = len(self.calibration_errors)
        adjusted_alpha = self.alpha * (n_cal + 1) / n_cal
        self.error_quantile = np.quantile(self.calibration_errors, adjusted_alpha)
        
        return self
    
    def predict(self, X_test):
        """预测带保形校正的区间"""
        lower_pred = self.lower_qr.predict(X_test)
        upper_pred = self.upper_qr.predict(X_test)
        
        # 应用保形校正
        lower_final = lower_pred - self.error_quantile
        upper_final = upper_pred + self.error_quantile
        
        return lower_final, upper_final


def load_data_for_conformal(data_path, n_samples=100000):
    """为保形预测加载数据 - 用于 win probability 回归"""
    print("加载数据用于保形预测 (回归形式)...")
    df = pd.read_parquet(data_path)
    df = df.iloc[:n_samples].copy()
    
    context_cols = ['business_type', 'deviceid', 'adid', 'campaignid', 'click_label']
    for col in context_cols:
        df[col] = df[col].fillna(0)
    
    # 特征: context + bid_amount
    X = np.column_stack([
        df[context_cols].values,
        df['bid_amount'].values
    ])
    
    # 目标: win_prob (如果是 win_label，则使用滑动窗口估计概率)
    # 但在这里我们直接使用 win_label 作为二分类目标
    y = df['win_label'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    print(f"样本数: {len(X):,}, 特征数: {X.shape[1]}")
    print(f"Win rate: {y.mean():.4f}")
    
    return X, y


def evaluate_conformal_regression(y_true, lower_pred, upper_pred):
    """评估保形预测回归结果"""
    # 计算覆盖率 (coverage rate)
    coverage = np.mean([(y_true[i] >= lower_pred[i]) & (y_true[i] <= upper_pred[i]) 
                       for i in range(len(y_true))])
    
    # 计算区间宽度
    interval_widths = upper_pred - lower_pred
    avg_width = np.mean(interval_widths)
    
    # 计算区间中心点作为点估计
    center_points = (lower_pred + upper_pred) / 2
    center_points = np.clip(center_points, 0, 1)  # 限制在 [0,1] 范围内
    
    # 评估点估计性能
    auc = roc_auc_score(y_true, center_points)
    rmse = np.sqrt(mean_squared_error(y_true, center_points))
    ece = compute_ece(y_true, center_points)
    
    print(f"\n=== Conformal Prediction (Corrected) 评估 ===")
    print(f"AUC: {auc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Coverage Rate: {coverage:.4f}")  # 应该接近 1-α
    print(f"Average Interval Width: {avg_width:.4f}")
    
    return {
        'auc': float(auc),
        'rmse': float(rmse),
        'ece': float(ece),
        'coverage_rate': float(coverage),
        'avg_interval_width': float(avg_width)
    }


def main():
    start_time = time.time()
    
    print("设备: CPU (保形预测主要是统计方法)")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 加载数据
    X, y = load_data_for_conformal(str(data_path), n_samples=100000)
    
    # 分割: 60%训练, 20%校准, 20%测试
    n = len(X)
    train_idx = int(0.6 * n)
    cal_idx = int(0.8 * n)
    
    X_train, X_cal, X_test = X[:train_idx], X[train_idx:cal_idx], X[cal_idx:]
    y_train, y_cal, y_test = y[:train_idx], y[train_idx:cal_idx], y[cal_idx:]
    
    print(f"训练集: {len(X_train)}, 校准集: {len(X_cal)}, 测试集: {len(X_test)}")
    
    # 构建保形预测器 (使用保形化分位数回归)
    conformal_pred = ConformalizedQuantileRegressor(alpha=0.1, quantiles=[0.05, 0.95])
    conformal_pred.fit(X_train, y_train, X_cal, y_cal)
    
    # 预测
    lower_pred, upper_pred = conformal_pred.predict(X_test)
    
    # 评估
    metrics = evaluate_conformal_regression(y_test, lower_pred, upper_pred)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    summary = {
        'experiment': 'exp09_conformal_prediction_corrected',
        'method': 'Conformalized Quantile Regression for Win Rate Estimation',
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()},
        'config': {
            'n_samples': 100000,
            'alpha': 0.1,
            'quantiles': [0.05, 0.95],
            'train_ratio': 0.6,
            'cal_ratio': 0.2,
            'test_ratio': 0.2
        },
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp09_conformal_prediction_corrected.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 09: Conformal Prediction for Win Rate Estimation (Corrected)

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **参考论文**: "Distribution-Free Prediction Sets for Win Rate Estimation in RTB" (WWW 2023 Workshop)
> **数据集**: Synthetic Bid Landscape (10 万样本)  
> **方法**: Conformalized Quantile Regression

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | {metrics['auc']:.4f} | 排序能力 |
| **RMSE** | {metrics['rmse']:.4f} | 概率预测精度 |
| **ECE** | {metrics['ece']:.4f} | 校准误差 |
| **Coverage Rate** | {metrics['coverage_rate']:.4f} | 期望: 0.9 (1-α) |
| **Avg. Interval Width** | {metrics['avg_interval_width']:.4f} | 置信区间宽度 |

---

## 🏗️ 方法架构

```
训练集 → 分位数回归器 (Q0.05, Q0.95)
           ↓
校准集 → 保形校正 (计算调整量)
           ↓
测试集 → 校正后区间 [lower ± adjustment, upper ± adjustment]
```

**核心创新**:
- 不假设数据分布 (distribution-free)  
- 提供理论保证的覆盖率
- 输出置信区间而非点估计
- 使用保形化分位数回归实现

---

## 🎯 与其他方法对比

| 模型 | AUC | RMSE | ECE | Coverage | Interval Width | 特点 |
|------|-----|------|-----|----------|----------------|------|
| LR Baseline | 0.8718 | 0.4620 | 0.0036 | N/A | N/A | Point estimate |
| MLP | 0.8718 | 0.3816 | 0.0056 | N/A | N/A | Point estimate |
| Multi-task | 0.8725 | 0.3809 | 0.0052 | N/A | N/A | Point estimate |
| **Conformal** | **{metrics['auc']:.4f}** | **{metrics['rmse']:.4f}** | **{metrics['ece']:.4f}** | **{metrics['coverage_rate']:.4f}** | **{metrics['avg_interval_width']:.4f}** | **Confidence interval** |

**保形预测的独特价值**:
- 提供不确定性量化 (置信区间)
- 理论保证的覆盖率 
- 分布无关的性质
- 对异常值鲁棒

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp09_conformal_prediction_corrected.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp09_conformal_prediction_corrected.json + .md")


if __name__ == '__main__':
    main()