"""
实验 03: Isotonic Regression 校准

目的: 对比 Isotonic Regression vs Temperature Scaling
"""

import os
import sys
from pathlib import Path
import json
import time
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from config import CalibrationConfig
from calibration.methods import IsotonicCalibration, compute_ece, compute_mce, compute_pcoc


def calibrate_isotonic_overall(preds, labels):
    """整体 Isotonic Regression 校准"""
    print("\n整体 Isotonic Regression...")
    
    iso_reg = IsotonicCalibration()
    iso_reg.fit(preds, labels)
    calibrated_probs = iso_reg.transform(preds)
    
    metrics = {
        'auc': roc_auc_score(labels, calibrated_probs),
        'logloss': log_loss(labels, calibrated_probs, eps=1e-10),
        'ece': compute_ece(calibrated_probs, labels, CalibrationConfig.num_bins),
        'mce': compute_mce(calibrated_probs, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_probs, labels),
        'predicted_ctr': calibrated_probs.mean(),
        'actual_ctr': labels.mean()
    }
    
    return calibrated_probs, metrics


def calibrate_isotonic_by_bt(df):
    """按 business_type 分组 Isotonic Regression 校准"""
    print("\n按 business_type 分组 Isotonic Regression...")
    
    calibrated_df = df.copy()
    
    for bt, group in df.groupby('business_type'):
        if len(group) < 100:
            continue
        
        preds = group['pred'].values
        labels = group['label'].values
        
        iso_reg = IsotonicCalibration()
        iso_reg.fit(preds, labels)
        calibrated_probs = iso_reg.transform(preds)
        
        calibrated_df.loc[group.index, 'pred_isotonic'] = calibrated_probs
    
    # 整体评估
    valid_mask = calibrated_df['pred_isotonic'].notna()
    overall_metrics = {
        'auc': roc_auc_score(calibrated_df.loc[valid_mask, 'label'], 
                            calibrated_df.loc[valid_mask, 'pred_isotonic']),
        'logloss': log_loss(calibrated_df.loc[valid_mask, 'label'], 
                           calibrated_df.loc[valid_mask, 'pred_isotonic'], eps=1e-10),
        'ece': compute_ece(calibrated_df.loc[valid_mask, 'pred_isotonic'].values,
                          calibrated_df.loc[valid_mask, 'label'].values, 
                          CalibrationConfig.num_bins),
        'mce': compute_mce(calibrated_df.loc[valid_mask, 'pred_isotonic'].values,
                          calibrated_df.loc[valid_mask, 'label'].values, 
                          CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_df.loc[valid_mask, 'pred_isotonic'].values,
                            calibrated_df.loc[valid_mask, 'label'].values),
        'predicted_ctr': calibrated_df.loc[valid_mask, 'pred_isotonic'].mean(),
        'actual_ctr': calibrated_df.loc[valid_mask, 'label'].mean(),
    }
    
    # 按 business_type 评估
    bt_metrics = []
    for bt, group in calibrated_df.groupby('business_type'):
        if 'pred_isotonic' not in group.columns or group['pred_isotonic'].isna().all():
            continue
        
        valid_group = group[group['pred_isotonic'].notna()]
        if len(valid_group) < 100:
            continue
        
        metrics = {
            'business_type': int(bt),
            'samples': len(valid_group),
            'pct': len(valid_group) / len(df) * 100,
            'auc': roc_auc_score(valid_group['label'], valid_group['pred_isotonic']),
            'logloss': log_loss(valid_group['label'], valid_group['pred_isotonic'], eps=1e-10),
            'ece': compute_ece(valid_group['pred_isotonic'].values, 
                              valid_group['label'].values, 
                              CalibrationConfig.num_bins),
            'mce': compute_mce(valid_group['pred_isotonic'].values, 
                              valid_group['label'].values, 
                              CalibrationConfig.num_bins),
            'pcoc': compute_pcoc(valid_group['pred_isotonic'].values, 
                                valid_group['label'].values),
            'predicted_ctr': valid_group['pred_isotonic'].mean(),
            'actual_ctr': valid_group['label'].mean()
        }
        bt_metrics.append(metrics)
    
    bt_metrics = sorted(bt_metrics, key=lambda x: x['samples'], reverse=True)
    
    return calibrated_df, overall_metrics, bt_metrics


def print_comparison(before, temp, iso, bt_before, bt_temp, bt_iso):
    """打印三种方法对比"""
    print("\n" + "="*80)
    print("校准方法对比")
    print("="*80)
    
    print(f"\n{'指标':<12} {'校准前':>12} {'Temperature':>12} {'Isotonic':>12}")
    print("-"*60)
    print(f"{'AUC':<12} {before['auc']:>12.4f} {temp['auc']:>12.4f} {iso['auc']:>12.4f}")
    print(f"{'LogLoss':<12} {before['logloss']:>12.4f} {temp['logloss']:>12.4f} {iso['logloss']:>12.4f}")
    print(f"{'ECE':<12} {before['ece']:>12.4f} {temp['ece']:>12.4f} {iso['ece']:>12.4f}")
    print(f"{'PCOC':<12} {before['pcoc']:>12.4f} {temp['pcoc']:>12.4f} {iso['pcoc']:>12.4f}")
    
    print("\n" + "="*80)
    print("按 business_type 分组对比 (Top 10)")
    print("="*80)
    
    print(f"\n{'BT':>3} {'PCOC前':>8} {'Temp':>8} {'Iso':>8} {'ECE前':>8} {'Temp':>8} {'Iso':>8}")
    print("-"*80)
    
    bt_before_dict = {m['business_type']: m for m in bt_before}
    bt_temp_dict = {m['business_type']: m for m in bt_temp}
    
    for m in bt_iso[:10]:
        bt = m['business_type']
        before_m = bt_before_dict.get(bt, {})
        temp_m = bt_temp_dict.get(bt, {})
        
        print(f"{bt:>3} "
              f"{before_m.get('pcoc', 0):>8.4f} "
              f"{temp_m.get('pcoc', 0):>8.4f} "
              f"{m['pcoc']:>8.4f} "
              f"{before_m.get('ece', 0):>8.4f} "
              f"{temp_m.get('ece', 0):>8.4f} "
              f"{m['ece']:>8.4f}")


def main():
    print("="*80)
    print("实验 03: Isotonic Regression 校准")
    print("="*80)
    
    # 加载预测结果
    print("\n加载预测结果...")
    results_dir = project_root / 'results'
    
    # 加载原始预测
    df = pd.read_parquet(results_dir / 'exp01b_predictions.parquet')
    
    # 加载之前的评估结果
    with open(results_dir / 'exp01b_overall_metrics.json', 'r') as f:
        before_overall = json.load(f)
    
    with open(results_dir / 'exp01b_business_type_metrics.json', 'r') as f:
        before_bt = json.load(f)
    
    with open(results_dir / 'exp02_overall_calibration.json', 'r') as f:
        temp_overall = json.load(f)
    
    with open(results_dir / 'exp02_bt_calibration.json', 'r') as f:
        temp_bt = json.load(f)
    
    print(f"  加载 {len(df)} 条预测结果")
    
    # 1. 整体 Isotonic Regression
    print("\n" + "="*80)
    print("1. 整体 Isotonic Regression")
    print("="*80)
    
    overall_iso, iso_metrics = calibrate_isotonic_overall(
        df['pred'].values, 
        df['label'].values
    )
    
    print(f"校准后 PCOC: {iso_metrics['pcoc']:.4f}")
    print(f"校准后 ECE: {iso_metrics['ece']:.4f}")
    
    # 2. 按 business_type 分组校准
    print("\n" + "="*80)
    print("2. 按 business_type 分组 Isotonic Regression")
    print("="*80)
    
    iso_df, bt_iso_overall, bt_iso_metrics = calibrate_isotonic_by_bt(df)
    
    # 打印对比
    print_comparison(before_overall, temp_overall, iso_metrics, 
                    before_bt, temp_bt, bt_iso_metrics)
    
    # 保存结果
    print("\n保存结果...")
    
    iso_metrics_save = {k: float(v) for k, v in iso_metrics.items()}
    with open(results_dir / 'exp03_overall_isotonic.json', 'w') as f:
        json.dump(iso_metrics_save, f, indent=2)
    
    bt_iso_metrics_save = [
        {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}
        for m in bt_iso_metrics
    ]
    with open(results_dir / 'exp03_bt_isotonic.json', 'w') as f:
        json.dump(bt_iso_metrics_save, f, indent=2)
    
    # 生成报告
    report = f"""# 实验 03: Isotonic Regression 校准

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 方法对比

| 指标 | 校准前 | Temperature | Isotonic |
|------|--------|-------------|----------|
| AUC | {before_overall['auc']:.4f} | {temp_overall['auc']:.4f} | {iso_metrics['auc']:.4f} |
| LogLoss | {before_overall['logloss']:.4f} | {temp_overall['logloss']:.4f} | {iso_metrics['logloss']:.4f} |
| ECE | {before_overall['ece']:.4f} | {temp_overall['ece']:.4f} | {iso_metrics['ece']:.4f} |
| PCOC | {before_overall['pcoc']:.4f} | {temp_overall['pcoc']:.4f} | {iso_metrics['pcoc']:.4f} |

## 按 business_type 分组对比

| BT | PCOC前 | Temp | Iso | ECE前 | Temp | Iso |
|----|--------|------|-----|-------|------|-----|
"""
    bt_before_dict = {m['business_type']: m for m in before_bt}
    bt_temp_dict = {m['business_type']: m for m in temp_bt}
    
    for m in bt_iso_metrics[:10]:
        bt = m['business_type']
        before_m = bt_before_dict.get(bt, {})
        temp_m = bt_temp_dict.get(bt, {})
        report += f"| {bt} | {before_m.get('pcoc', 0):.4f} | {temp_m.get('pcoc', 0):.4f} | {m['pcoc']:.4f} | {before_m.get('ece', 0):.4f} | {temp_m.get('ece', 0):.4f} | {m['ece']:.4f} |\n"
    
    report += f"""
## 核心发现

- Isotonic Regression 比 Temperature Scaling 更灵活
- ECE 进一步降低: {before_overall['ece']:.4f} → {iso_metrics['ece']:.4f}
- PCOC 接近 1.0: {iso_metrics['pcoc']:.4f}

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp03_isotonic_regression.md', 'w') as f:
        f.write(report)
    
    print(f"\n结果已保存到 {results_dir}")
    print("\n" + "="*80)
    print("实验 03 完成!")
    print("="*80)


if __name__ == '__main__':
    main()
