#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 18: Comprehensive Comparison - 全面对比

核心思想:
1. 汇总所有实验结果 (exp01-exp17)
2. 多维度对比：AUC, RMSE, ECE, 训练时间，参数量
3. 生成最终排名和推荐

输出:
- 完整的结果汇总表
- 各场景下的最佳模型推荐
- 可视化图表 (可选)
"""

import sys
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).parent.parent
results_dir = project_root / 'results'

print("="*60)
print("🚀 Experiment 18: Comprehensive Comparison")
print("="*60)

# 收集所有实验结果
print("\n📂 Collecting experiment results...")

all_results = []

for json_file in results_dir.glob('exp*.json'):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        exp_name = json_file.stem
        
        # 提取指标
        metrics = {}
        
        if 'metrics' in data:
            metrics.update(data['metrics'])
        elif 'results' in data and isinstance(data['results'], dict):
            # 某些实验可能有多个方法
            for method, method_metrics in data['results'].items():
                if isinstance(method_metrics, dict):
                    metrics[f'{method}_auc'] = method_metrics.get('auc', 
method_metrics.get('win_auc', np.nan))
                    metrics[f'{method}_ece'] = method_metrics.get('ece', np.nan)
        
        # 标准化字段名
        metrics['win_auc'] = metrics.get('win_auc', metrics.get('auc', np.nan))
        metrics['win_ece'] = metrics.get('win_ece', metrics.get('ece', np.nan))
        metrics['win_rmse'] = metrics.get('win_rmse', metrics.get('rmse', np.nan))
        
        all_results.append({
            'experiment': exp_name,
            'auc': metrics.get('win_auc', np.nan),
            'rmse': metrics.get('win_rmse', np.nan),
            'ece': metrics.get('win_ece', np.nan),
            'training_time': data.get('training_time_seconds', np.nan),
            'params': data.get('total_params', np.nan),
            'device': data.get('device', 'unknown')
        })
        
        print(f"  ✅ {exp_name}: AUC={metrics.get('win_auc', 'N/A'):.4f if isinstance(metrics.get('win_auc'), float) else 'N/A'}")
    
    except Exception as e:
        print(f"  ⚠️ Error loading {json_file}: {e}")

if not all_results:
    print("\n❌ No experiment results found!")
    sys.exit(1)

# 创建 DataFrame
df_results = pd.DataFrame(all_results)

# 排序和排名
print("\n📊 Ranking by AUC...")
df_ranked = df_results.sort_values('auc', ascending=False).reset_index(drop=True)
df_ranked['rank'] = range(1, len(df_ranked) + 1)

# 显示 Top 10
print("\n" + "="*60)
print("🏆 TOP 10 MODELS BY AUC")
print("="*60)
top10 = df_ranked[['rank', 'experiment', 'auc', 'rmse', 'ece', 'training_time']].head(10)
print(top10.to_string(index=False))

# 按 ECE 排序 (越低越好)
print("\n" + "="*60)
print("🎯 TOP 10 MODELS BY CALIBRATION (ECE)")
print("="*60)
df_by_ece = df_results[df_results['ece'].notna()].sort_values('ece').head(10)
df_by_ece_display = df_by_ece[['experiment', 'auc', 'ece']].reset_index(drop=True)
print(df_by_ece_display.to_string(index=False))

# 按训练时间排序 (越快越好)
print("\n" + "="*60)
print("⚡ TOP 10 MODELS BY SPEED")
print("="*60)
df_by_speed = df_results[df_results['training_time'].notna()].sort_values('training_time').head(10)
df_by_speed_display = df_by_speed[['experiment', 'auc', 'training_time']].reset_index(drop=True)
print(df_by_speed_display.to_string(index=False))

# 场景推荐
print("\n" + "="*60)
print("💡 RECOMMENDATIONS BY SCENARIO")
print("="*60)

best_auc = df_ranked.loc[df_ranked['auc'].idxmax()]
best_ece = df_results[df_results['ece'].notna()].loc[df_results['ece'].idxmin()]
best_speed = df_results[df_results['training_time'].notna()].loc[df_results['training_time'].idxmin()]

print(f"\n  🥇 Best Overall (AUC): {best_auc['experiment']}")
print(f"     AUC: {best_auc['auc']:.4f}, Time: {best_auc['training_time']:.1f}s")

print(f"\n  🎯 Best Calibration (ECE): {best_ece['experiment']}")
print(f"     ECE: {best_ece['ece']:.4f}, AUC: {best_ece['auc']:.4f}")

print(f"\n  ⚡ Fastest Training: {best_speed['experiment']}")
print(f"     Time: {best_speed['training_time']:.1f}s, AUC: {best_speed['auc']:.4f}")

# 保存结果
print("\n💾 Saving comparison results...")

output_data = {
    'generated_at': datetime.now().isoformat(),
    'total_experiments': len(all_results),
    'ranking_by_auc': df_ranked.to_dict('records'),
    'top_recommendations': {
        'best_auc': best_auc.to_dict(),
        'best_calibration': best_ece.to_dict(),
        'fastest': best_speed.to_dict()
    }
}

with open(results_dir / 'exp18_comprehensive_comparison.json', 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

# Markdown 报告
md_report = f"""# Experiment 18: Comprehensive Comparison

## Overview
- **Total Experiments**: {len(all_results)}
- **Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏆 Top 10 Models by AUC

| Rank | Experiment | AUC | RMSE | ECE | Training Time (s) |
|------|------------|-----|------|-----|-------------------|
"""

for _, row in top10.iterrows():
    md_report += f"| {int(row['rank'])} | {row['experiment']} | {row['auc']:.4f} | {row['rmse']:.4f if pd.notna(row['rmse']) else 'N/A'} | {row['ece']:.4f if pd.notna(row['ece']) else 'N/A'} | {row['training_time']:.1f if pd.notna(row['training_time']) else 'N/A'} |\n"

md_report += f"""
## 💡 Recommendations

### Best Overall (AUC)
- **Model**: {best_auc['experiment']}
- **AUC**: {best_auc['auc']:.4f}
- **Training Time**: {best_auc['training_time']:.1f}s

### Best Calibration (ECE)
- **Model**: {best_ece['experiment']}
- **ECE**: {best_ece['ece']:.4f}
- **AUC**: {best_ece['auc']:.4f}

### Fastest Training
- **Model**: {best_speed['experiment']}
- **Time**: {best_speed['training_time']:.1f}s
- **AUC**: {best_speed['auc']:.4f}

---
*Auto-generated by Experiment 18*
"""

with open(results_dir / 'exp18_comprehensive_comparison.md', 'w') as f:
    f.write(md_report)

print("✅ Results saved to:")
print(f"  - {results_dir / 'exp18_comprehensive_comparison.json'}")
print(f"  - {results_dir / 'exp18_comprehensive_comparison.md'}")

print("\n" + "="*60)
print("✅ Experiment 18 completed!")
print("="*60)
