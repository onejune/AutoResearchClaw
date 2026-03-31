"""
实验 11: Two-Stage Calibration (Temperature + Isotonic)

目的：结合 Temperature Scaling 和 Isotonic Regression，先粗略校准再精细校准
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
from config import CalibrationConfig
from calibration.methods import compute_ece, compute_mce, compute_pcoc


class TemperatureScaling:
    """Temperature Scaling"""
    
    def __init__(self):
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, logits):
        return torch.sigmoid(logits / self.temperature)
    
    def fit(self, logits, labels):
        """学习温度参数"""
        logits = torch.tensor(logits, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
        
        def closure():
            optimizer.zero_grad()
            probs = self.forward(logits)
            loss = nn.functional.binary_cross_entropy(probs, labels)
            loss.backward()
            return loss
        
        for _ in range(50):
            optimizer.step(closure)
        
        return self
    
    def transform(self, preds):
        """应用校准"""
        eps = 1e-10
        probs = np.clip(preds, eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        
        with torch.no_grad():
            calibrated = self.forward(torch.tensor(logits, dtype=torch.float32)).numpy()
        
        return calibrated


class TwoStageCalibration:
    """Two-Stage Calibration: Temperature + Isotonic"""
    
    def __init__(self):
        self.temp_scaling = TemperatureScaling()
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, preds, labels):
        """拟合"""
        # Stage 1: Temperature Scaling
        eps = 1e-10
        probs = np.clip(preds, eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        
        self.temp_scaling.fit(logits, labels)
        
        # Stage 2: Isotonic Regression on temperature-scaled predictions
        temp_calibrated = self.temp_scaling.transform(preds)
        self.iso_reg.fit(temp_calibrated, labels)
        
        return self
    
    def transform(self, preds):
        """应用校准"""
        # Stage 1
        temp_calibrated = self.temp_scaling.transform(preds)
        # Stage 2
        final_calibrated = self.iso_reg.predict(temp_calibrated)
        
        return np.clip(final_calibrated, 1e-7, 1-1e-7)


def main():
    print("="*80)
    print("实验 11: Two-Stage Calibration (Temperature + Isotonic)")
    print("="*80)
    
    # 加载数据
    results_dir = project_root / 'results'
    df = pd.read_parquet(results_dir / 'exp01b_predictions.parquet')
    
    preds = df['pred'].values
    labels = df['label'].values
    
    print(f"\n数据：{len(preds)} 样本")
    
    # 1. Baseline
    print("\n1. Baseline...")
    metrics_baseline = {
        'auc': roc_auc_score(labels, preds),
        'logloss': log_loss(labels, preds, eps=1e-10),
        'ece': compute_ece(preds, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(preds, labels)
    }
    print(f"Baseline: AUC={metrics_baseline['auc']:.4f}, ECE={metrics_baseline['ece']:.6f}, PCOC={metrics_baseline['pcoc']:.4f}")
    
    # 2. Temperature Scaling only
    print("\n2. Temperature Scaling...")
    temp_only = TemperatureScaling()
    eps = 1e-10
    probs = np.clip(preds, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    temp_only.fit(logits, labels)
    calibrated_temp = temp_only.transform(preds)
    
    metrics_temp = {
        'auc': roc_auc_score(labels, calibrated_temp),
        'logloss': log_loss(labels, calibrated_temp, eps=1e-10),
        'ece': compute_ece(calibrated_temp, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_temp, labels),
        'temperature': temp_only.temperature.item()
    }
    print(f"Temp Only: T={metrics_temp['temperature']:.4f}, AUC={metrics_temp['auc']:.4f}, ECE={metrics_temp['ece']:.6f}, PCOC={metrics_temp['pcoc']:.4f}")
    
    # 3. Isotonic only
    print("\n3. Isotonic Regression...")
    iso_only = IsotonicRegression(out_of_bounds='clip')
    iso_only.fit(preds, labels)
    calibrated_iso = iso_only.predict(preds)
    
    metrics_iso = {
        'auc': roc_auc_score(labels, calibrated_iso),
        'logloss': log_loss(labels, calibrated_iso, eps=1e-10),
        'ece': compute_ece(calibrated_iso, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_iso, labels)
    }
    print(f"Iso Only: AUC={metrics_iso['auc']:.4f}, ECE={metrics_iso['ece']:.6f}, PCOC={metrics_iso['pcoc']:.4f}")
    
    # 4. Two-Stage
    print("\n4. Two-Stage Calibration...")
    two_stage = TwoStageCalibration()
    two_stage.fit(preds, labels)
    calibrated_two = two_stage.transform(preds)
    
    metrics_two = {
        'auc': roc_auc_score(labels, calibrated_two),
        'logloss': log_loss(labels, calibrated_two, eps=1e-10),
        'ece': compute_ece(calibrated_two, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_two, labels),
        'temperature': two_stage.temp_scaling.temperature.item()
    }
    print(f"Two-Stage: T={metrics_two['temperature']:.4f}, AUC={metrics_two['auc']:.4f}, ECE={metrics_two['ece']:.6f}, PCOC={metrics_two['pcoc']:.4f}")
    
    # 对比
    print("\n" + "="*80)
    print("方法对比")
    print("="*80)
    print(f"{'Method':>20} {'AUC':>10} {'ECE':>12} {'PCOC':>10}")
    print("-"*60)
    methods = [
        ("Baseline", metrics_baseline),
        ("Temperature Only", metrics_temp),
        ("Isotonic Only", metrics_iso),
        ("Two-Stage", metrics_two)
    ]
    for name, m in methods:
        print(f"{name:>20} {m['auc']:>10.4f} {m['ece']:>12.6f} {m['pcoc']:>10.4f}")
    
    # 保存结果
    with open(results_dir / 'exp11_two_stage.json', 'w') as f:
        json.dump({
            'baseline': metrics_baseline,
            'temp_only': metrics_temp,
            'iso_only': metrics_iso,
            'two_stage': metrics_two
        }, f, indent=2)
    
    # 生成报告
    report = f"""# 实验 11: Two-Stage Calibration

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 方法对比

| Method | AUC | ECE | PCOC | Temperature |
|--------|-----|-----|------|-------------|
| Baseline | {metrics_baseline['auc']:.4f} | {metrics_baseline['ece']:.6f} | {metrics_baseline['pcoc']:.4f} | - |
| Temperature Only | {metrics_temp['auc']:.4f} | {metrics_temp['ece']:.6f} | {metrics_temp['pcoc']:.4f} | {metrics_temp['temperature']:.4f} |
| Isotonic Only | {metrics_iso['auc']:.4f} | {metrics_iso['ece']:.6f} | {metrics_iso['pcoc']:.4f} | - |
| Two-Stage | {metrics_two['auc']:.4f} | {metrics_two['ece']:.6f} | {metrics_two['pcoc']:.4f} | {metrics_two['temperature']:.4f} |

## 核心发现

- Two-Stage vs Iso Only: {('改善' if metrics_two['ece'] < metrics_iso['ece'] else '基本不变')}
- Temperature 参数：{metrics_two['temperature']:.4f}
- 两步校准的优势：{('更稳定' if abs(metrics_two['temperature'] - 1.0) < abs(metrics_temp['temperature'] - 1.0) else '无明显优势')}

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp11_two_stage.md', 'w') as f:
        f.write(report)
    
    print(f"\n✅ 结果已保存到 {results_dir}")


if __name__ == '__main__':
    main()
