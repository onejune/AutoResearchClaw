"""
实验 01B: 基线模型校准性能评估（按 business_type 分组）

目的: 评估不同 business_type 的校准性能
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
from calibration.methods import compute_ece, compute_mce, compute_pcoc
from training.hardware_monitor import select_training_device


def evaluate_by_business_type(model, loader, device, vocab_sizes):
    """按 business_type 分组评估"""
    model.eval()
    
    # 找到 business_type 的索引
    feature_cols = get_feature_cols()
    bt_idx = feature_cols.index('business_type')
    
    # 收集所有预测和标签
    all_data = []
    
    with torch.no_grad():
        for features, labels in loader:
            features_device, labels_device = features.to(device), labels.to(device)
            logits = model(features_device)
            probs = torch.sigmoid(logits)
            
            # 获取 business_type
            business_types = features[:, bt_idx].cpu().numpy()
            
            for i in range(len(labels)):
                all_data.append({
                    'business_type': business_types[i],
                    'pred': probs[i].item(),
                    'label': labels[i].item()
                })
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_data)
    
    # 按整体评估
    overall_metrics = {
        'auc': roc_auc_score(df['label'], df['pred']),
        'logloss': log_loss(df['label'], df['pred'], eps=1e-10),
        'ece': compute_ece(df['pred'].values, df['label'].values, CalibrationConfig.num_bins),
        'mce': compute_mce(df['pred'].values, df['label'].values, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(df['pred'].values, df['label'].values),
        'predicted_ctr': df['pred'].mean(),
        'actual_ctr': df['label'].mean(),
        'samples': len(df)
    }
    
    # 按 business_type 分组评估
    bt_metrics = []
    bt_groups = df.groupby('business_type')
    
    for bt, group in bt_groups:
        if len(group) < 100:  # 样本太少跳过
            continue
        
        try:
            metrics = {
                'business_type': int(bt),
                'samples': len(group),
                'pct': len(group) / len(df) * 100,
                'auc': roc_auc_score(group['label'], group['pred']),
                'logloss': log_loss(group['label'], group['pred'], eps=1e-10),
                'ece': compute_ece(group['pred'].values, group['label'].values, CalibrationConfig.num_bins),
                'mce': compute_mce(group['pred'].values, group['label'].values, CalibrationConfig.num_bins),
                'pcoc': compute_pcoc(group['pred'].values, group['label'].values),
                'predicted_ctr': group['pred'].mean(),
                'actual_ctr': group['label'].mean()
            }
            bt_metrics.append(metrics)
        except Exception as e:
            print(f"  Warning: business_type {bt} evaluation failed: {e}")
    
    # 按样本量排序
    bt_metrics = sorted(bt_metrics, key=lambda x: x['samples'], reverse=True)
    
    return overall_metrics, bt_metrics, df


def print_metrics_by_bt(overall, bt_metrics, top_k=10):
    """打印按 business_type 分组的结果"""
    print("\n" + "="*80)
    print("整体评估")
    print("="*80)
    print(f"  AUC:    {overall['auc']:.4f}")
    print(f"  LogLoss: {overall['logloss']:.4f}")
    print(f"  ECE:    {overall['ece']:.4f}")
    print(f"  MCE:    {overall['mce']:.4f}")
    print(f"  PCOC:   {overall['pcoc']:.4f}")
    print(f"  样本数: {overall['samples']:,}")
    
    print("\n" + "="*80)
    print(f"按 business_type 分组评估 (Top {top_k})")
    print("="*80)
    print(f"{'BT':>4} {'样本数':>8} {'占比%':>6} {'AUC':>6} {'ECE':>6} {'PCOC':>6} {'预测CTR':>8} {'实际CTR':>8}")
    print("-"*80)
    
    for i, m in enumerate(bt_metrics[:top_k]):
        print(f"{m['business_type']:>4} "
              f"{m['samples']:>8,} "
              f"{m['pct']:>6.2f} "
              f"{m['auc']:>6.4f} "
              f"{m['ece']:>6.4f} "
              f"{m['pcoc']:>6.4f} "
              f"{m['predicted_ctr']:>8.4f} "
              f"{m['actual_ctr']:>8.4f}")
    
    # 统计信息
    print("\n" + "="*80)
    print("统计信息")
    print("="*80)
    aucs = [m['auc'] for m in bt_metrics]
    eces = [m['ece'] for m in bt_metrics]
    pcocs = [m['pcoc'] for m in bt_metrics]
    
    print(f"AUC 范围: [{min(aucs):.4f}, {max(aucs):.4f}]")
    print(f"ECE 范围: [{min(eces):.4f}, {max(eces):.4f}]")
    print(f"PCOC 范围: [{min(pcocs):.4f}, {max(pcocs):.4f}]")
    
    # 找出校准差的 business_type
    bad_calibration = [m for m in bt_metrics if m['pcoc'] < 0.8 or m['pcoc'] > 1.2]
    if bad_calibration:
        print(f"\n⚠️  校准偏差大的 business_type (PCOC < 0.8 或 > 1.2):")
        for m in bad_calibration[:5]:
            print(f"  BT {m['business_type']}: PCOC={m['pcoc']:.4f}, AUC={m['auc']:.4f}, 样本数={m['samples']:,}")


def main():
    print("="*80)
    print("实验 01B: 基线模型校准性能评估（按 business_type 分组）")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(ExperimentConfig.seed)
    np.random.seed(ExperimentConfig.seed)
    
    # 动态选择设备
    device = select_training_device(min_memory_mb=4096, max_utilization=90.0)
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    vocab_sizes = load_vocab_sizes()
    feature_cols = get_feature_cols()
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        batch_size=TrainConfig.batch_size,
        num_workers=TrainConfig.num_workers
    )
    
    # 加载训练好的模型
    print("\n加载模型...")
    model = MLPCtr(
        vocab_sizes=vocab_sizes,
        feature_cols=feature_cols,
        embed_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    ).to(device)
    
    checkpoint_path = project_root / 'checkpoints' / 'exp01_baseline_best.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"已加载模型: {checkpoint_path}")
    else:
        print(f"⚠️  模型文件不存在，将使用未训练的模型")
    
    # 在测试集上按 business_type 评估
    print("\n评估中...")
    overall_metrics, bt_metrics, df = evaluate_by_business_type(model, test_loader, device, vocab_sizes)
    
    # 打印结果
    print_metrics_by_bt(overall_metrics, bt_metrics, top_k=10)
    
    # 保存结果
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 保存整体指标
    overall_metrics_serializable = {k: float(v) for k, v in overall_metrics.items()}
    with open(results_dir / 'exp01b_overall_metrics.json', 'w') as f:
        json.dump(overall_metrics_serializable, f, indent=2)
    
    # 保存分组指标
    bt_metrics_serializable = [
        {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}
        for m in bt_metrics
    ]
    with open(results_dir / 'exp01b_business_type_metrics.json', 'w') as f:
        json.dump(bt_metrics_serializable, f, indent=2)
    
    # 保存详细数据
    df.to_parquet(results_dir / 'exp01b_predictions.parquet', index=False)
    
    # 生成报告
    report = f"""# 实验 01B: 基线模型校准性能评估（按 business_type 分组）

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}
> **设备**: {device}

## 整体评估

| 指标 | 值 |
|------|-----|
| AUC | {overall_metrics['auc']:.4f} |
| LogLoss | {overall_metrics['logloss']:.4f} |
| ECE | {overall_metrics['ece']:.4f} |
| MCE | {overall_metrics['mce']:.4f} |
| PCOC | {overall_metrics['pcoc']:.4f} |
| 样本数 | {overall_metrics['samples']:,} |

## 按 business_type 分组评估 (Top 10)

| BT | 样本数 | 占比% | AUC | ECE | PCOC | 预测CTR | 实际CTR |
|----|--------|-------|-----|-----|------|---------|---------|
"""
    for m in bt_metrics[:10]:
        report += f"| {m['business_type']} | {m['samples']:,} | {m['pct']:.2f} | {m['auc']:.4f} | {m['ece']:.4f} | {m['pcoc']:.4f} | {m['predicted_ctr']:.4f} | {m['actual_ctr']:.4f} |\n"
    
    aucs = [m['auc'] for m in bt_metrics]
    eces = [m['ece'] for m in bt_metrics]
    pcocs = [m['pcoc'] for m in bt_metrics]
    
    report += f"""
## 统计信息

- AUC 范围: [{min(aucs):.4f}, {max(aucs):.4f}]
- ECE 范围: [{min(eces):.4f}, {max(eces):.4f}]
- PCOC 范围: [{min(pcocs):.4f}, {max(pcocs):.4f}]

## 文件

- 整体指标: `exp01b_overall_metrics.json`
- 分组指标: `exp01b_business_type_metrics.json`
- 详细数据: `exp01b_predictions.parquet`

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp01b_business_type_evaluation.md', 'w') as f:
        f.write(report)
    
    print(f"\n结果已保存到 {results_dir}")
    print("\n" + "="*80)
    print("实验 01B 完成!")
    print("="*80)


if __name__ == '__main__':
    main()
