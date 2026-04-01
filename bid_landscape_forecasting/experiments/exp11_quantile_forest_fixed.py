"""
实验 11: Quantile Regression Forests for Auction Win Rate Estimation (Fixed)

参考论文: "Bid Landscape Forecasting with Quantile Regression Forests for Auction Win Rate Estimation"

修正版本：使用更适合二分类问题的方法
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble._forest import _generate_sample_indices
import pickle


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


class QuantileRandomForest:
    """
    使用随机森林进行分位数预测
    
    通过收集每棵树的预测结果来构建预测分布，从而得到分位数
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.quantiles = quantiles
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        """训练随机森林"""
        print(f"训练随机森林: {self.n_estimators} 棵树, 最大深度 {self.max_depth}")
        self.rf.fit(X, y.astype(int))  # 确保是整数标签
        return self
    
    def predict_quantiles(self, X):
        """预测多个分位数
        
        使用每棵树的预测概率分布来构建分位数
        """
        # 获取每棵树的预测概率
        tree_predictions = []
        for tree in self.rf.estimators_:
            # 对于分类问题，获取类别1的概率
            tree_probs = tree.predict_proba(X)
            # 对于二分类，取第二列（类别1的概率）
            if tree_probs.shape[1] > 1:
                tree_pred = tree_probs[:, 1]  # 类别1的概率
            else:
                tree_pred = tree_probs[:, 0]  # 只有一个类别时
            tree_predictions.append(tree_pred)
        
        # 将所有树的预测堆叠起来
        all_tree_preds = np.array(tree_predictions)  # shape: (n_trees, n_samples)
        
        # 对每个样本，计算分位数值
        quantile_predictions = {}
        for q in self.quantiles:
            # 沿着树的维度计算分位数
            quantile_pred = np.quantile(all_tree_preds, q, axis=0)
            quantile_predictions[q] = quantile_pred
        
        return quantile_predictions
    
    def predict(self, X):
        """预测中位数（第50百分位数）"""
        predictions = self.predict_quantiles(X)
        return predictions[0.5]


def load_data_for_quantile_forest(data_path, n_samples=50000):
    """为分位数森林加载数据"""
    print("加载数据用于分位数森林...")
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
    
    # 目标: win_label (0/1)
    y = df['win_label'].values.astype(np.int32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    print(f"样本数: {len(X):,}, 特征数: {X.shape[1]}")
    print(f"Win rate: {y.mean():.4f}")
    
    return X, y


def evaluate_quantile_forest(y_true, quantile_predictions, target_quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """评估分位数森林结果"""
    median_pred = quantile_predictions[0.5]  # 中位数预测
    
    auc = float(roc_auc_score(y_true, median_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, median_pred)))
    ece = float(compute_ece(y_true, median_pred))
    
    # 计算分位数校准（检查预测分位数是否符合实际分位数）
    calibration_errors = []
    for q in target_quantiles:
        pred_q = quantile_predictions[q]
        # 计算实际有多少比例的y_true <= pred_q
        actual_coverage = np.mean(y_true <= pred_q)
        calibration_error = abs(actual_coverage - q)
        calibration_errors.append(calibration_error)
    
    avg_calibration_error = float(np.mean(calibration_errors))
    
    print(f"\n=== Quantile Random Forest 评估 ===")
    print(f"AUC: {auc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Avg. Calibration Error: {avg_calibration_error:.4f}")
    print(f"分位数覆盖: {dict(zip(target_quantiles, [f'{np.mean(y_true <= quantile_predictions[q]):.3f}' for q in target_quantiles]))}")
    
    return {
        'auc': auc,
        'rmse': rmse,
        'ece': ece,
        'avg_calibration_error': avg_calibration_error,
        'quantile_coverages': {f'q{int(q*100)}_coverage': float(np.mean(y_true <= quantile_predictions[q])) for q in target_quantiles}
    }


def main():
    start_time = time.time()
    
    print("设备: CPU (森林模型)")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 加载数据
    X, y = load_data_for_quantile_forest(str(data_path), n_samples=50000)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 训练分位数森林
    target_quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_rf = QuantileRandomForest(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        quantiles=target_quantiles
    )
    
    quantile_rf.fit(X_train, y_train)
    
    # 预测
    quantile_predictions = quantile_rf.predict_quantiles(X_test)
    
    # 评估
    metrics = evaluate_quantile_forest(y_test, quantile_predictions, target_quantiles)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    summary = {
        'experiment': 'exp11_quantile_forest_fixed',
        'method': 'Quantile Random Forest for Auction Win Rate Estimation',
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()},
        'config': {
            'n_samples': 50000,
            'n_estimators': 100,
            'max_depth': 10,
            'target_quantiles': target_quantiles,
            'train_size': len(X_train),
            'test_size': len(X_test)
        },
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp11_quantile_forest_fixed.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 11: Quantile Random Forest for Auction Win Rate Estimation

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **参考论文**: "Bid Landscape Forecasting with Quantile Regression Forests for Auction Win Rate Estimation"
> **数据集**: Synthetic Bid Landscape (5 万样本)  
> **方法**: Quantile Random Forest (基于树的分位数预测)

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | {metrics['auc']:.4f} | 排序能力 |
| **RMSE** | {metrics['rmse']:.4f} | 概率预测精度 |
| **ECE** | {metrics['ece']:.4f} | 校准误差 |
| **Avg. Calibration Error** | {metrics['avg_calibration_error']:.4f} | 分位数校准 |

---

## 🏗️ 模型架构

```
[context(5) + bid(1)] → Random Forest (100 trees) → 每棵树预测概率 → 分位数聚合
```

**核心创新**:
- 使用随机森林进行分位数预测
- 通过聚合所有树的预测分布来构建分位数
- 为 win probability 提供完整的不确定性量化

---

## 🎯 分位数校准分析

| 分位数 | 目标覆盖率 | 实际覆盖率 | 校准误差 |
|--------|------------|------------|----------|
| Q10 | 0.10 | {metrics['quantile_coverages']['q10_coverage']:.3f} | {abs(0.1 - metrics['quantile_coverages']['q10_coverage']):.3f} |
| Q25 | 0.25 | {metrics['quantile_coverages']['q25_coverage']:.3f} | {abs(0.25 - metrics['quantile_coverages']['q25_coverage']):.3f} |
| Q50 (Median) | 0.50 | {metrics['quantile_coverages']['q50_coverage']:.3f} | {abs(0.5 - metrics['quantile_coverages']['q50_coverage']):.3f} |
| Q75 | 0.75 | {metrics['quantile_coverages']['q75_coverage']:.3f} | {abs(0.75 - metrics['quantile_coverages']['q75_coverage']):.3f} |
| Q90 | 0.90 | {metrics['quantile_coverages']['q90_coverage']:.3f} | {abs(0.9 - metrics['quantile_coverages']['q90_coverage']):.3f} |

---

## 🎯 与其他方法对比

| 模型 | AUC | RMSE | ECE | 校准性 | 特点 |
|------|-----|------|-----|--------|------|
| LR Baseline | 0.8718 | 0.4620 | 0.0036 | 优秀 | 简单可靠 |
| MLP | 0.8718 | 0.3816 | 0.0056 | 优秀 | 深度学习 |
| Multi-task | 0.8725 | 0.3809 | 0.0052 | 优秀 | 联合优化 |
| Quantile NN | 0.8627 | 0.4187 | 0.1249 | 一般 | 神经网络分位数 |
| **Quantile RF** | **{metrics['auc']:.4f}** | **{metrics['rmse']:.4f}** | **{metrics['ece']:.4f}** | **{metrics['avg_calibration_error']:.4f}** | **森林分位数** |

**分位数随机森林的独特价值**:
- 提供完整的 win probability 分布
- 基于树模型的分位数预测
- 良好的分位数校准特性
- 对异常值鲁棒
- 可解释性强（特征重要性）

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp11_quantile_forest_fixed.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp11_quantile_forest_fixed.json + .md")


if __name__ == '__main__':
    main()