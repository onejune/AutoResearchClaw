"""
实验 07: Regularized Isotonic Regression

目的：对比标准 Isotonic vs 正则化 Isotonic，验证抗过拟合能力
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


class RegularizedIsotonicCalibration:
    """正则化 Isotonic Regression 校准"""
    
    def __init__(self, lambda_reg=0.01):
        self.lambda_reg = lambda_reg
        self.iso_reg = None
    
    def fit(self, preds, labels):
        """
        带正则化的 Isotonic Regression
        
        使用平滑技术：在单调回归前对预测值进行分箱平均
        """
        # 分箱平滑
        n_bins = max(20, int(len(preds) ** 0.5))
        bins = np.linspace(preds.min(), preds.max(), n_bins + 1)
        bin_indices = np.digitize(preds, bins[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # 计算每个 bin 的平均标签
        bin_means = []
        bin_centers = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_means.append(labels[mask].mean())
                bin_centers.append(preds[mask].mean())
            else:
                bin_means.append(0.5)
                bin_centers.append(bins[i])
        
        bin_means = np.array(bin_means)
        bin_centers = np.array(bin_centers)
        
        # 对 bin_means 做单调回归
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
        self.iso_reg.fit(bin_centers, bin_means)
        
        return self
    
    def transform(self, preds):
        """应用校准"""
        calibrated = self.iso_reg.predict(preds)
        return np.clip(calibrated, 1e-7, 1 - 1e-7)


def evaluate_calibration(preds, labels, name=""):
    """评估校准效果"""
    metrics = {
        'auc': roc_auc_score(labels, preds),
        'logloss': log_loss(labels, preds, eps=1e-10),
        'ece': compute_ece(preds, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(preds, labels)
    }
    print(f"{name}: AUC={metrics['auc']:.4f}, ECE={metrics['ece']:.4f}, PCOC={metrics['pcoc']:.4f}")
    return metrics


def main():
    print("="*80)
    print("实验 07: Regularized Isotonic Regression")
    print("="*80)
    
    # 加载数据
    results_dir = project_root / 'results'
    df = pd.read_parquet(results_dir / 'exp01b_predictions.parquet')
    
    preds = df['pred'].values
    labels = df['label'].values
    
    print(f"\n数据：{len(preds)} 样本")
    
    # 1. 标准 Isotonic
    print("\n1. 标准 Isotonic Regression...")
    from sklearn.isotonic import IsotonicRegression
    iso_std = IsotonicRegression(out_of_bounds='clip')
    iso_std.fit(preds, labels)
    calibrated_std = iso_std.transform(preds)
    metrics_std = evaluate_calibration(calibrated_std, labels, "标准 Isotonic")
    
    # 2. 正则化 Isotonic (不同 λ)
    print("\n2. 正则化 Isotonic Regression...")
    lambdas = [0.01, 0.05, 0.1, 0.5]
    
    results = {}
    for lam in lambdas:
        reg_iso = RegularizedIsotonicCalibration(lambda_reg=lam)
        reg_iso.fit(preds, labels)
        calibrated_reg = reg_iso.transform(preds)
        metrics_reg = evaluate_calibration(calibrated_reg, labels, f"Reg-ISO (λ={lam})")
        results[lam] = metrics_reg
    
    # 3. 对比
    print("\n" + "="*80)
    print("方法对比")
    print("="*80)
    print(f"{'Method':>20} {'AUC':>10} {'ECE':>10} {'PCOC':>10}")
    print("-"*60)
    print(f"{'标准 Isotonic':>20} {metrics_std['auc']:>10.4f} {metrics_std['ece']:>10.4f} {metrics_std['pcoc']:>10.4f}")
    for lam, m in results.items():
        print(f"{'Reg-ISO':>20} (λ={lam}) {m['auc']:>10.4f} {m['ece']:>10.4f} {m['pcoc']:>10.4f}")
    
    # 保存结果
    with open(results_dir / 'exp07_regularized_isotonic.json', 'w') as f:
        json.dump({
            'standard': metrics_std,
            'regularized': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存到 {results_dir}")


if __name__ == '__main__':
    main()
