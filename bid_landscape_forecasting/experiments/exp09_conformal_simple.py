"""
实验 09: Simple Conformal Prediction for Win Rate Estimation (Simplified Implementation)

参考论文: "Distribution-Free Prediction Sets for Win Rate Estimation in RTB" (WWW 2023 Workshop)

使用更简单的保形预测方法 - 基于残差的保形预测
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


class ResidualConformalPredictor:
    """
    基于残差的保形预测器 (Simple Split Conformal Prediction)
    
    实现步骤:
    1. 使用训练集训练基础预测器
    2. 在校准集上计算残差的分位数
    3. 对新样本预测，加上校正的置信区间
    """
    
    def __init__(self, base_model, alpha=0.1):
        self.base_model = base_model  # 基础预测模型 (例如 LogisticRegression)
        self.alpha = alpha  # 错误容忍度
        self.error_quantile = None  # 残差分位数
    
    def fit(self, X_train, y_train, X_cal, y_cal):
        """训练基础模型并在校准集上计算保形校正"""
        # 训练基础模型
        self.base_model.fit(X_train, y_train)
        
        # 在校准集上预测
        if hasattr(self.base_model, 'predict_proba'):
            # 对于分类器，获取 win probability
            cal_probs = self.base_model.predict_proba(X_cal)[:, 1]
        else:
            # 对于回归器
            cal_probs = self.base_model.predict(X_cal)
        
        # 计算残差 (绝对误差)
        calibration_residuals = np.abs(y_cal - cal_probs)
        
        # 计算 (1-α) 分位数
        n_cal = len(calibration_residuals)
        # 使用保形预测的标准公式
        adjusted_alpha = (1 - self.alpha) * (1 + 1/n_cal)  # 1 - α + 1/n
        self.error_quantile = np.quantile(calibration_residuals, adjusted_alpha)
        
        return self
    
    def predict_interval(self, X_test):
        """预测置信区间"""
        # 基础预测
        if hasattr(self.base_model, 'predict_proba'):
            test_probs = self.base_model.predict_proba(X_test)[:, 1]
        else:
            test_probs = self.base_model.predict(X_test)
        
        # 添加保形校正
        lower_bounds = np.clip(test_probs - self.error_quantile, 0, 1)
        upper_bounds = np.clip(test_probs + self.error_quantile, 0, 1)
        
        return test_probs, lower_bounds, upper_bounds


def load_data_for_conformal(data_path, n_samples=100000):
    """为保形预测加载数据"""
    print("加载数据用于保形预测...")
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
    y = df['win_label'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    print(f"样本数: {len(X):,}, 特征数: {X.shape[1]}")
    print(f"Win rate: {y.mean():.4f}")
    
    return X, y


def evaluate_conformal_prediction(y_true, y_pred_center, lower_bounds, upper_bounds):
    """评估保形预测结果"""
    # 计算覆盖率 (coverage rate)
    coverage = np.mean([(y_true[i] >= lower_bounds[i]) & (y_true[i] <= upper_bounds[i]) 
                       for i in range(len(y_true))])
    
    # 计算区间宽度
    interval_widths = upper_bounds - lower_bounds
    avg_width = np.mean(interval_widths)
    
    # 评估中心点预测性能
    auc = roc_auc_score(y_true, y_pred_center)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_center))
    ece = compute_ece(y_true, y_pred_center)
    
    print(f"\n=== Simple Conformal Prediction 评估 ===")
    print(f"AUC: {auc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Coverage Rate: {coverage:.4f}")  # 应该接近 1-α
    print(f"Average Interval Width: {avg_width:.4f}")
    print(f"Conformity Bandwidth: ±{np.unique(np.round(upper_bounds - y_pred_center, 4))[0]:.4f}")
    
    return {
        'auc': float(auc),
        'rmse': float(rmse),
        'ece': float(ece),
        'coverage_rate': float(coverage),
        'avg_interval_width': float(avg_width),
        'bandwidth': float(np.unique(upper_bounds - y_pred_center)[0])
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
    
    # 创建基础模型
    base_model = LogisticRegression(max_iter=1000)
    
    # 构建保形预测器
    conformal_pred = ResidualConformalPredictor(base_model, alpha=0.1)  # 90% 置信区间
    conformal_pred.fit(X_train, y_train, X_cal, y_cal)
    
    # 预测
    y_pred_center, lower_pred, upper_pred = conformal_pred.predict_interval(X_test)
    
    # 评估
    metrics = evaluate_conformal_prediction(y_test, y_pred_center, lower_pred, upper_pred)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    summary = {
        'experiment': 'exp09_conformal_simple',
        'method': 'Residual-Based Conformal Prediction for Win Rate Estimation',
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()},
        'config': {
            'n_samples': 100000,
            'alpha': 0.1,  # 90% 置信水平
            'base_model': 'LogisticRegression',
            'train_ratio': 0.6,
            'cal_ratio': 0.2,
            'test_ratio': 0.2
        },
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp09_conformal_simple.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 09: Simple Conformal Prediction for Win Rate Estimation

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **参考论文**: "Distribution-Free Prediction Sets for Win Rate Estimation in RTB" (WWW 2023 Workshop)
> **数据集**: Synthetic Bid Landscape (10 万样本)  
> **方法**: Residual-Based Split Conformal Prediction

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | {metrics['auc']:.4f} | 排序能力 |
| **RMSE** | {metrics['rmse']:.4f} | 概率预测精度 |
| **ECE** | {metrics['ece']:.4f} | 校准误差 |
| **Coverage Rate** | {metrics['coverage_rate']:.4f} | 期望: 0.9 (1-α) |
| **Avg. Interval Width** | {metrics['avg_interval_width']:.4f} | 置信区间宽度 |
| **Bandwidth** | ±{metrics['bandwidth']:.4f} | 置信带宽度 |

---

## 🏗️ 方法架构

```
训练集 → 基础模型 (Logistic Regression)
           ↓
校准集 → 残差分析 (计算 (1-α) 分位数)
           ↓
测试集 → 预测 + 保形校正 → [pred - ε, pred + ε]
```

**核心创新**:
- 不假设数据分布 (distribution-free)  
- 提供理论保证的覆盖率
- 输出置信区间而非点估计
- 基于残差的简单保形预测

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
- 理论保证的覆盖率 (目标 90%)
- 分布无关的性质
- 对异常值鲁棒

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp09_conformal_simple.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp09_conformal_simple.json + .md")


if __name__ == '__main__':
    main()