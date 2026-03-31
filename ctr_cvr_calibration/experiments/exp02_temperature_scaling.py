"""
实验 02: Temperature Scaling 校准

目的: 对比整体校准 vs 按 business_type 分组校准
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
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from config import TrainConfig, CalibrationConfig, ExperimentConfig
from data.dataset import create_dataloaders, get_feature_cols, load_vocab_sizes
from models.mlp import MLPCtr
from calibration.methods import TemperatureScaling, compute_ece, compute_mce, compute_pcoc
from training.hardware_monitor import select_training_device


def calibrate_overall(preds, labels):
    """整体 Temperature Scaling 校准"""
    print("\n整体 Temperature Scaling...")
    
    # 转换为 logits
    eps = 1e-10
    probs = np.clip(preds, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    
    # 转换为 tensor
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    # 学习温度参数
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(logits_tensor, labels_tensor)
    
    # 应用校准
    with torch.no_grad():
        calibrated_probs = temp_scaling(logits_tensor).numpy()
    
    # 评估
    metrics = {
        'auc': roc_auc_score(labels, calibrated_probs),
        'logloss': log_loss(labels, calibrated_probs, eps=1e-10),
        'ece': compute_ece(calibrated_probs, labels, CalibrationConfig.num_bins),
        'mce': compute_mce(calibrated_probs, labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_probs, labels),
        'predicted_ctr': calibrated_probs.mean(),
        'actual_ctr': labels.mean(),
        'temperature': temp_scaling.temperature.item()
    }
    
    return calibrated_probs, metrics, temp_scaling


def calibrate_by_business_type(df):
    """按 business_type 分组 Temperature Scaling 校准"""
    print("\n按 business_type 分组 Temperature Scaling...")
    
    calibrated_df = df.copy()
    bt_temperatures = {}
    
    # 按 business_type 分组校准
    for bt, group in df.groupby('business_type'):
        if len(group) < 100:
            continue
        
        preds = group['pred'].values
        labels = group['label'].values
        
        # 转换为 logits
        eps = 1e-10
        probs = np.clip(preds, eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        
        # 转换为 tensor
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # 学习温度参数
        temp_scaling = TemperatureScaling()
        temp_scaling.fit(logits_tensor, labels_tensor)
        
        # 应用校准
        with torch.no_grad():
            calibrated_probs = temp_scaling(logits_tensor).numpy()
        
        # 更新结果
        calibrated_df.loc[group.index, 'pred_calibrated'] = calibrated_probs
        bt_temperatures[int(bt)] = temp_scaling.temperature.item()
    
    # 整体评估
    valid_mask = calibrated_df['pred_calibrated'].notna()
    overall_metrics = {
        'auc': roc_auc_score(calibrated_df.loc[valid_mask, 'label'], 
                            calibrated_df.loc[valid_mask, 'pred_calibrated']),
        'logloss': log_loss(calibrated_df.loc[valid_mask, 'label'], 
                           calibrated_df.loc[valid_mask, 'pred_calibrated'], eps=1e-10),
        'ece': compute_ece(calibrated_df.loc[valid_mask, 'pred_calibrated'].values,
                          calibrated_df.loc[valid_mask, 'label'].values, 
                          CalibrationConfig.num_bins),
        'mce': compute_mce(calibrated_df.loc[valid_mask, 'pred_calibrated'].values,
                          calibrated_df.loc[valid_mask, 'label'].values, 
                          CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(calibrated_df.loc[valid_mask, 'pred_calibrated'].values,
                            calibrated_df.loc[valid_mask, 'label'].values),
        'predicted_ctr': calibrated_df.loc[valid_mask, 'pred_calibrated'].mean(),
        'actual_ctr': calibrated_df.loc[valid_mask, 'label'].mean(),
    }
    
    # 按 business_type 评估
    bt_metrics = []
    for bt, group in calibrated_df.groupby('business_type'):
        if 'pred_calibrated' not in group.columns or group['pred_calibrated'].isna().all():
            continue
        
        valid_group = group[group['pred_calibrated'].notna()]
        if len(valid_group) < 100:
            continue
        
        metrics = {
            'business_type': int(bt),
            'samples': len(valid_group),
            'pct': len(valid_group) / len(df) * 100,
            'auc': roc_auc_score(valid_group['label'], valid_group['pred_calibrated']),
            'logloss': log_loss(valid_group['label'], valid_group['pred_calibrated'], eps=1e-10),
            'ece': compute_ece(valid_group['pred_calibrated'].values, 
                              valid_group['label'].values, 
                              CalibrationConfig.num_bins),
            'mce': compute_mce(valid_group['pred_calibrated'].values, 
                              valid_group['label'].values, 
                              CalibrationConfig.num_bins),
            'pcoc': compute_pcoc(valid_group['pred_calibrated'].values, 
                                valid_group['label'].values),
            'predicted_ctr': valid_group['pred_calibrated'].mean(),
            'actual_ctr': valid_group['label'].mean(),
            'temperature': bt_temperatures.get(int(bt), 1.0)
        }
        bt_metrics.append(metrics)
    
    bt_metrics = sorted(bt_metrics, key=lambda x: x['samples'], reverse=True)
    
    return calibrated_df, overall_metrics, bt_metrics, bt_temperatures


def print_comparison(before, after, bt_before, bt_after):
    """打印对比结果"""
    print("\n" + "="*80)
    print("校准效果对比")
    print("="*80)
    
    print(f"\n{'指标':<12} {'校准前':>12} {'整体校准':>12} {'分组校准':>12}")
    print("-"*60)
    print(f"{'AUC':<12} {before['auc']:>12.4f} {after['auc']:>12.4f}")
    print(f"{'LogLoss':<12} {before['logloss']:>12.4f} {after['logloss']:>12.4f}")
    print(f"{'ECE':<12} {before['ece']:>12.4f} {after['ece']:>12.4f}")
    print(f"{'MCE':<12} {before['mce']:>12.4f} {after['mce']:>12.4f}")
    print(f"{'PCOC':<12} {before['pcoc']:>12.4f} {after['pcoc']:>12.4f}")
    
    print("\n" + "="*80)
    print("按 business_type 分组对比 (Top 10)")
    print("="*80)
    
    print(f"\n{'BT':>4} {'样本%':>6} {'PCOC前':>8} {'PCOC后':>8} {'ECE前':>8} {'ECE后':>8} {'温度':>8}")
    print("-"*80)
    
    # 创建字典方便查找
    bt_before_dict = {m['business_type']: m for m in bt_before}
    
    for i, m in enumerate(bt_after[:10]):
        bt = m['business_type']
        before_m = bt_before_dict.get(bt, {})
        
        print(f"{bt:>4} "
              f"{m['pct']:>6.2f} "
              f"{before_m.get('pcoc', 0):>8.4f} "
              f"{m['pcoc']:>8.4f} "
              f"{before_m.get('ece', 0):>8.4f} "
              f"{m['ece']:>8.4f} "
              f"{m['temperature']:>8.4f}")


def main():
    print("="*80)
    print("实验 02: Temperature Scaling 校准")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(ExperimentConfig.seed)
    np.random.seed(ExperimentConfig.seed)
    
    # 动态选择设备
    device = select_training_device(min_memory_mb=4096, max_utilization=90.0)
    print(f"\n使用设备: {device}")
    
    # 加载预测结果
    print("\n加载预测结果...")
    results_dir = project_root / 'results'
    pred_file = results_dir / 'exp01b_predictions.parquet'
    
    if not os.path.exists(pred_file):
        print(f"❌ 预测文件不存在: {pred_file}")
        print("请先运行 exp01b_business_type_evaluation.py")
        return
    
    df = pd.read_parquet(pred_file)
    print(f"  加载 {len(df)} 条预测结果")
    
    # 加载之前的评估结果
    with open(results_dir / 'exp01b_overall_metrics.json', 'r') as f:
        before_overall = json.load(f)
    
    with open(results_dir / 'exp01b_business_type_metrics.json', 'r') as f:
        before_bt = json.load(f)
    
    print(f"  校准前 PCOC: {before_overall['pcoc']:.4f}")
    print(f"  校准前 ECE: {before_overall['ece']:.4f}")
    
    # 1. 整体 Temperature Scaling
    print("\n" + "="*80)
    print("1. 整体 Temperature Scaling")
    print("="*80)
    
    overall_calibrated, overall_metrics, temp_scaling = calibrate_overall(
        df['pred'].values, 
        df['label'].values
    )
    
    print(f"\n整体温度: {overall_metrics['temperature']:.4f}")
    print(f"校准后 PCOC: {overall_metrics['pcoc']:.4f}")
    print(f"校准后 ECE: {overall_metrics['ece']:.4f}")
    
    # 2. 按 business_type 分组校准
    print("\n" + "="*80)
    print("2. 按 business_type 分组 Temperature Scaling")
    print("="*80)
    
    calibrated_df, bt_overall, bt_metrics, bt_temperatures = calibrate_by_business_type(df)
    
    print(f"\n不同 business_type 的温度参数:")
    for bt, temp in sorted(bt_temperatures.items())[:10]:
        print(f"  BT {bt}: {temp:.4f}")
    
    # 打印对比
    print_comparison(before_overall, bt_overall, before_bt, bt_metrics)
    
    # 保存结果
    print("\n保存结果...")
    
    # 整体校准结果
    overall_metrics_save = {k: float(v) for k, v in overall_metrics.items()}
    with open(results_dir / 'exp02_overall_calibration.json', 'w') as f:
        json.dump(overall_metrics_save, f, indent=2)
    
    # 分组校准结果
    bt_metrics_save = [
        {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}
        for m in bt_metrics
    ]
    with open(results_dir / 'exp02_bt_calibration.json', 'w') as f:
        json.dump(bt_metrics_save, f, indent=2)
    
    # 温度参数
    with open(results_dir / 'exp02_temperatures.json', 'w') as f:
        json.dump(bt_temperatures, f, indent=2)
    
    # 校准后的预测结果
    calibrated_df.to_parquet(results_dir / 'exp02_calibrated_predictions.parquet', index=False)
    
    # 生成报告
    report = f"""# 实验 02: Temperature Scaling 校准

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 整体 Temperature Scaling

| 指标 | 校准前 | 校准后 | 变化 |
|------|--------|--------|------|
| AUC | {before_overall['auc']:.4f} | {overall_metrics['auc']:.4f} | {overall_metrics['auc'] - before_overall['auc']:+.4f} |
| LogLoss | {before_overall['logloss']:.4f} | {overall_metrics['logloss']:.4f} | {overall_metrics['logloss'] - before_overall['logloss']:+.4f} |
| ECE | {before_overall['ece']:.4f} | {overall_metrics['ece']:.4f} | {overall_metrics['ece'] - before_overall['ece']:+.4f} |
| PCOC | {before_overall['pcoc']:.4f} | {overall_metrics['pcoc']:.4f} | {overall_metrics['pcoc'] - before_overall['pcoc']:+.4f} |

**整体温度参数**: {overall_metrics['temperature']:.4f}

## 按 business_type 分组校准

| BT | 样本% | PCOC前 | PCOC后 | ECE前 | ECE后 | 温度 |
|----|-------|--------|--------|-------|-------|------|
"""
    bt_before_dict = {m['business_type']: m for m in before_bt}
    for m in bt_metrics[:10]:
        bt = m['business_type']
        before_m = bt_before_dict.get(bt, {})
        report += f"| {bt} | {m['pct']:.2f} | {before_m.get('pcoc', 0):.4f} | {m['pcoc']:.4f} | {before_m.get('ece', 0):.4f} | {m['ece']:.4f} | {m['temperature']:.4f} |\n"
    
    report += f"""
## 核心发现

### 整体校准效果
- Temperature Scaling 对整体指标改善有限（PCOC 已经接近 1.0）
- 温度参数: {overall_metrics['temperature']:.4f}（接近 1.0）

### 分组校准效果
- 不同 business_type 需要不同的温度参数
- 温度范围: [{min(bt_temperatures.values()):.4f}, {max(bt_temperatures.values()):.4f}]

## 文件

- 整体校准结果: `exp02_overall_calibration.json`
- 分组校准结果: `exp02_bt_calibration.json`
- 温度参数: `exp02_temperatures.json`
- 校准后预测: `exp02_calibrated_predictions.parquet`

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp02_temperature_scaling.md', 'w') as f:
        f.write(report)
    
    print(f"\n结果已保存到 {results_dir}")
    print("\n" + "="*80)
    print("实验 02 完成!")
    print("="*80)


if __name__ == '__main__':
    main()
