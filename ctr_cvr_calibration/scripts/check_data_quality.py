#!/usr/bin/env python3
"""
数据质量检查脚本

按照实验规范，所有模型实验前必须执行数据质量检查
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


def check_data_quality(data_path):
    """
    数据质量检查
    
    检查项目:
    1. 样本数量 & 正样本率（CTR/CVR/CTCVR）
    2. 特征缺失率
    3. 特征 vocab 分布
    4. 标签分布是否合理
    """
    print("="*60)
    print("数据质量检查")
    print("="*60)
    
    # 加载元数据
    meta_path = os.path.join(data_path, 'meta.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print(f"\n📊 基本信息:")
    print(f"  特征数量: {meta['n_features']}")
    print(f"  标签列: {meta['label_cols']}")
    
    # 加载词表大小
    vocab_path = os.path.join(data_path, 'vocab_sizes.json')
    with open(vocab_path, 'r') as f:
        vocab_sizes = json.load(f)
    
    print(f"\n📚 词表大小统计:")
    vocab_values = list(vocab_sizes.values())
    print(f"  最小: {min(vocab_values):,}")
    print(f"  最大: {max(vocab_values):,}")
    print(f"  平均: {np.mean(vocab_values):,.0f}")
    print(f"  中位数: {np.median(vocab_values):,.0f}")
    
    # 加载训练数据
    train_path = os.path.join(data_path, 'train/')
    print(f"\n📂 加载训练数据...")
    df = pd.read_parquet(train_path)
    
    print(f"\n📊 数据集统计:")
    print(f"  样本数: {len(df):,}")
    
    # 标签分布
    print(f"\n🎯 标签分布:")
    for label_col in meta['label_cols']:
        label_mean = df[label_col].mean()
        label_sum = df[label_col].sum()
        print(f"  {label_col}:")
        print(f"    正样本率: {label_mean:.4f} ({label_mean*100:.2f}%)")
        print(f"    正样本数: {label_sum:,}")
    
    # 特征缺失率
    print(f"\n❓ 特征缺失率:")
    feature_cols = meta['feature_cols']
    missing_rates = df[feature_cols].isnull().mean()
    missing_features = missing_rates[missing_rates > 0]
    if len(missing_features) > 0:
        print(f"  有缺失的特征 ({len(missing_features)} 个):")
        for col, rate in missing_features.items():
            print(f"    {col}: {rate:.4f}")
    else:
        print(f"  ✅ 无缺失值")
    
    # 特征分布
    print(f"\n📈 特征分布 (前 10 个):")
    for col in feature_cols[:10]:
        col_min = df[col].min()
        col_max = df[col].max()
        col_unique = df[col].nunique()
        expected_vocab = vocab_sizes.get(col, 'N/A')
        print(f"  {col}:")
        print(f"    范围: [{col_min:,}, {col_max:,}]")
        print(f"    唯一值: {col_unique:,}")
        print(f"    词表大小: {expected_vocab}")
    
    # 标签合理性检查
    print(f"\n✅ 标签合理性检查:")
    ctr = df['click_label'].mean()
    ctcvr = df['ctcvr_label'].mean()
    
    warnings = []
    
    # CTR 过高/过低
    if ctr > 0.7:
        warnings.append(f"⚠️ CTR 过高 ({ctr:.2%})，可能存在数据偏差")
    elif ctr < 0.01:
        warnings.append(f"⚠️ CTR 过低 ({ctr:.2%})，可能存在数据偏差")
    else:
        print(f"  ✅ CTR 合理 ({ctr:.2%})")
    
    # CTCVR 应该 <= CTR
    if ctcvr > ctr:
        warnings.append(f"⚠️ CTCVR ({ctcvr:.2%}) > CTR ({ctr:.2%})，逻辑异常")
    else:
        print(f"  ✅ CTCVR <= CTR")
    
    # CVR 估算
    cvr = ctcvr / ctr if ctr > 0 else 0
    print(f"  估算 CVR (CTCVR/CTR): {cvr:.2%}")
    
    if cvr > 0.5:
        warnings.append(f"⚠️ CVR 过高 ({cvr:.2%})，可能存在数据偏差")
    
    # 输出警告
    if warnings:
        print(f"\n⚠️ 警告:")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\n✅ 数据质量检查通过")
    
    # 保存检查报告
    report = {
        'dataset': 'IVR Sample v16 CTCVR',
        'samples': len(df),
        'features': meta['n_features'],
        'labels': {
            col: {
                'mean': float(df[col].mean()),
                'sum': int(df[col].sum())
            }
            for col in meta['label_cols']
        },
        'vocab_sizes': {
            'min': int(min(vocab_values)),
            'max': int(max(vocab_values)),
            'mean': float(np.mean(vocab_values)),
            'median': float(np.median(vocab_values))
        },
        'missing_rate': {
            col: float(rate)
            for col, rate in missing_rates.items()
            if rate > 0
        },
        'warnings': warnings
    }
    
    report_path = Path(__file__).parent.parent / 'results' / 'data_quality_report.json'
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 检查报告已保存: {report_path}")
    
    return report


if __name__ == '__main__':
    data_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/'
    check_data_quality(data_path)
