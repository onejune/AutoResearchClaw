"""
实验 09: Quantile Isotonic Regression (分位数校准)

目的：对不同置信度区间分别校准，高置信度区域更精确
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
from config import CalibrationConfig
from calibration.methods import compute_ece, compute_pcoc


class QuantileIsotonicCalibration:
    """
    分位数 Isotonic Calibration
    
    将预测值分成几个区间，每个区间独立做单调回归
    """
    
    def __init__(self, quantiles=[0.1, 0.3, 0.7]):
        self.quantiles = quantiles
        self.calibrators = {}
        self.bounds = None
    
    def fit(self, preds, labels):
        """拟合"""
        # 计算分位数边界
        self.bounds = np.percentile(preds, [0] + [q*100 for q in self.quantiles] + [100])
        
        # 为每个区间训练独立的 Isotonic Regression
        for i in range(len(self.bounds) - 1):
            lower, upper = self.bounds[i], self.bounds[i+1]
            mask = (preds >= lower) & (preds < upper) if i < len(self.bounds) - 2 else (preds >= lower) & (preds <= upper)
            
            if mask.sum() > 100:  # 至少 100 个样本
                self.calibrators[i] = IsotonicRegression(out_of_bounds='clip')
                self.calibrators[i].fit(preds[mask], labels[mask])
        
        return self
    
    def transform(self, preds):
        """应用校准"""
        calibrated = np.zeros_like(preds)
        
        for i in range(len(self.bounds) - 1):
            lower, upper = self.bounds[i], self.bounds[i+1]
            mask = (preds >= lower) & (preds < upper) if i < len(self.bounds) - 2 else (preds >= lower) & (preds <= upper)
            
            if i in self.calibrators:
                calibrated[mask] = self.calibrators[i].predict(preds[mask])
            else:
                calibrated[mask] = preds[mask]  # 没有足够数据，保持原样
        
        return np.clip(calibrated, 1e-7, 1-1e-7)


def evaluate_by_quantile(preds, labels, quantiles=[0.1, 0.3, 0.7]):
    """按分位数评估"""
    bounds = np.percentile(preds, [0] + [q*100 for q in quantiles] + [100])
    
    results = []
    for i in range(len(bounds) - 1):
        lower, upper = bounds[i], bounds[i+1]
        mask = (preds >= lower) & (preds < upper) if i < len(bounds) - 2 else (preds >= lower) & (preds <= upper)
        
        if mask.sum() > 100:
            p, l = preds[mask], labels[mask]
            results.append({
                'range': f"[{lower:.3f}, {upper:.3f})",
                'samples': mask.sum(),
                'auc': roc_auc_score(l, p),
                'ece': compute_ece(p, l, CalibrationConfig.num_bins),
                'pcoc': compute_pcoc(p, l)
            })
    
    return results


def main():
    print("="*80)
    print("实验 09: Quantile Isotonic Regression")
    print("="*80)
    
    # 加载数据
    results_dir = project_root / 'results'
    df = pd.read_parquet(results_dir / 'exp01b_predictions.parquet')
    
    preds = df['pred'].values
    labels = df['label'].values
    
    print(f"\n数据：{len(preds)} 样本")
    
    # 1. 整体 Isotonic
    print("\n1. 整体 Isotonic Regression...")
    iso_std = IsotonicRegression(out_of_bounds='clip')
    iso_std.fit(preds, labels)
    calibrated_std = iso_std.predict(preds)
    
    metrics_std = {
        'auc': roc_auc_score(labels, calibrated_std),
        'logloss': log_loss(labels, calibrated_std, eps=1e-10),
        'ece': compute_ece(calibrated_std, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_std, labels)
    }
    print(f"整体：AUC={metrics_std['auc']:.4f}, ECE={metrics_std['ece']:.6f}, PCOC={metrics_std['pcoc']:.4f}")
    
    # 2. Quantile Isotonic (不同分位数配置)
    quantile_configs = [
        [0.5],           # 2 个区间
        [0.3, 0.7],      # 3 个区间
        [0.2, 0.4, 0.6, 0.8]  # 5 个区间
    ]
    
    results = {}
    for quantiles in quantile_configs:
        name = f"Q-{len(quantiles)+1}bin"
        print(f"\n2.{name}: {quantiles}...")
        
        q_iso = QuantileIsotonicCalibration(quantiles=quantiles)
        q_iso.fit(preds, labels)
        calibrated_q = q_iso.transform(preds)
        
        metrics_q = {
            'auc': roc_auc_score(labels, calibrated_q),
            'logloss': log_loss(labels, calibrated_q, eps=1e-10),
            'ece': compute_ece(calibrated_q, labels, CalibrationConfig.num_bins),
            'pcoc': compute_pcoc(calibrated_q, labels)
        }
        print(f"  整体：AUC={metrics_q['auc']:.4f}, ECE={metrics_q['ece']:.6f}, PCOC={metrics_q['pcoc']:.4f}")
        
        # 按区间评估
        interval_results = evaluate_by_quantile(calibrated_q, labels, quantiles)
        for r in interval_results[:3]:  # 只打印前 3 个区间
            print(f"  {r['range']:>20} samples={r['samples']:>6} AUC={r['auc']:.4f} ECE={r['ece']:.6f}")
        
        results[name] = metrics_q
    
    # 3. 对比
    print("\n" + "="*80)
    print("方法对比")
    print("="*80)
    print(f"{'Method':>15} {'AUC':>10} {'ECE':>12} {'PCOC':>10}")
    print("-"*50)
    print(f"{'整体 Isotonic':>15} {metrics_std['auc']:>10.4f} {metrics_std['ece']:>12.6f} {metrics_std['pcoc']:>10.4f}")
    for name, m in results.items():
        print(f"{name:>15} {m['auc']:>10.4f} {m['ece']:>12.6f} {m['pcoc']:>10.4f}")
    
    # 保存结果
    with open(results_dir / 'exp09_quantile_isotonic.json', 'w') as f:
        json.dump({
            'standard': metrics_std,
            'quantile': results
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到 {results_dir}")


if __name__ == '__main__':
    main()
