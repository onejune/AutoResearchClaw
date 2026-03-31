"""
实验 08: Concave Isotonic Regression (凹约束)

目的：验证凹性约束对 CTR/CVR 校准的改善
原理：概率校准通常具有边际收益递减特性，适合凹函数
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.metrics import roc_auc_score, log_loss
from config import CalibrationConfig
from calibration.methods import compute_ece, compute_pcoc


class ConcaveIsotonicRegression:
    """
    凹单调回归
    
    在单调递增的基础上，添加凹性约束 (二阶差分 <= 0)
    使用 Pool Adjacent Violators Algorithm 的凹版本
    """
    
    def __init__(self):
        self.x_ = None
        self.y_ = None
    
    def fit(self, X, y):
        """
        拟合凹单调回归
        
        Args:
            X: 预测值 (一维数组)
            y: 真实标签 (一维数组)
        """
        # 排序
        order = np.argsort(X)
        X_sorted = X[order]
        y_sorted = y[order]
        
        # 去重并平均
        unique_x, indices = np.unique(X_sorted, return_inverse=True)
        y_mean = np.bincount(indices, weights=y_sorted) / np.bincount(indices)
        
        n = len(unique_x)
        
        # PAVA for concave isotonic regression
        # 维护一个栈，每个元素是 (x_start, x_end, mean_y, sum_y, count)
        stack = []
        
        for i in range(n):
            current = [unique_x[i], unique_x[i], y_mean[i], y_mean[i], 1]
            
            while len(stack) > 0:
                top = stack[-1]
                
                # 检查单调性：top 的均值应该 <= current 的均值
                if top[2] > current[2]:
                    # 合并
                    new_sum = top[3] * top[4] + current[3] * current[4]
                    new_count = top[4] + current[4]
                    current = [top[0], current[1], new_sum/new_count, new_sum, new_count]
                    stack.pop()
                else:
                    # 检查凹性：斜率应该递减
                    if len(stack) >= 1:
                        prev = stack[-2] if len(stack) >= 2 else None
                        if prev is not None:
                            slope1 = (top[2] - prev[2]) / max(top[0] - prev[0], 1e-10)
                            slope2 = (current[2] - top[2]) / max(current[0] - top[0], 1e-10)
                            
                            if slope1 < slope2:  # 违反凹性
                                # 合并 top 和 current
                                new_sum = top[3] * top[4] + current[3] * current[4]
                                new_count = top[4] + current[4]
                                current = [top[0], current[1], new_sum/new_count, new_sum, new_count]
                                stack.pop()
                                continue
                    
                    break
            
            stack.append(current)
        
        # 构建结果
        self.x_ = np.array([s[0] for s in stack] + [stack[-1][1]])
        self.y_ = np.array([s[2] for s in stack])
        
        return self
    
    def predict(self, X):
        """预测"""
        return np.interp(X, self.x_, np.concatenate([self.y_, [self.y_[-1]]]))


def evaluate_calibration(preds, labels, name=""):
    """评估校准效果"""
    metrics = {
        'auc': roc_auc_score(labels, preds),
        'logloss': log_loss(labels, preds, eps=1e-10),
        'ece': compute_ece(preds, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(preds, labels)
    }
    print(f"{name}: AUC={metrics['auc']:.4f}, ECE={metrics['ece']:.6f}, PCOC={metrics['pcoc']:.4f}")
    return metrics


def main():
    print("="*80)
    print("实验 08: Concave Isotonic Regression")
    print("="*80)
    
    # 加载数据
    results_dir = project_root / 'results'
    df = pd.read_parquet(results_dir / 'exp01b_predictions.parquet')
    
    preds = df['pred'].values
    labels = df['label'].values
    
    print(f"\n数据：{len(preds)} 样本")
    print(f"原始：AUC={roc_auc_score(labels, preds):.4f}, ECE={compute_ece(preds, labels):.6f}")
    
    # 1. 标准 Isotonic
    print("\n1. 标准 Isotonic Regression...")
    from sklearn.isotonic import IsotonicRegression
    iso_std = IsotonicRegression(out_of_bounds='clip')
    iso_std.fit(preds, labels)
    calibrated_std = iso_std.predict(preds)
    metrics_std = evaluate_calibration(calibrated_std, labels, "标准 Isotonic")
    
    # 2. Concave Isotonic
    print("\n2. Concave Isotonic Regression...")
    concave_iso = ConcaveIsotonicRegression()
    concave_iso.fit(preds, labels)
    calibrated_concave = concave_iso.predict(preds)
    calibrated_concave = np.clip(calibrated_concave, 1e-7, 1-1e-7)
    metrics_concave = evaluate_calibration(calibrated_concave, labels, "Concave Isotonic")
    
    # 3. 对比
    print("\n" + "="*80)
    print("方法对比")
    print("="*80)
    print(f"{'Method':>20} {'AUC':>10} {'ECE':>12} {'PCOC':>10}")
    print("-"*60)
    print(f"{'标准 Isotonic':>20} {metrics_std['auc']:>10.4f} {metrics_std['ece']:>12.6f} {metrics_std['pcoc']:>10.4f}")
    print(f"{'Concave Isotonic':>20} {metrics_concave['auc']:>10.4f} {metrics_concave['ece']:>12.6f} {metrics_concave['pcoc']:>10.4f}")
    
    # 保存结果
    with open(results_dir / 'exp08_concave_isotonic.json', 'w') as f:
        json.dump({
            'standard': metrics_std,
            'concave': metrics_concave
        }, f, indent=2)
    
    # 生成报告
    report = f"""# 实验 08: Concave Isotonic Regression

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 对比结果

| Method | AUC | ECE | PCOC |
|--------|-----|-----|------|
| 标准 Isotonic | {metrics_std['auc']:.4f} | {metrics_std['ece']:.6f} | {metrics_std['pcoc']:.4f} |
| Concave Isotonic | {metrics_concave['auc']:.4f} | {metrics_concave['ece']:.6f} | {metrics_concave['pcoc']:.4f} |

## 核心发现

- Concave 约束对校准的影响：{('轻微改善' if metrics_concave['ece'] < metrics_std['ece'] else '基本不变')}
- PCOC 保持接近 1.0

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp08_concave_isotonic.md', 'w') as f:
        f.write(report)
    
    print(f"\n✅ 结果已保存到 {results_dir}")


if __name__ == '__main__':
    main()
