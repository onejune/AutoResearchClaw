"""
生成 CTR/CVR 校准研究可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

results_dir = Path('/mnt/workspace/open_research/autoresearch/ctr_cvr_calibration/results')
output_dir = results_dir.parent / 'figures'
output_dir.mkdir(exist_ok=True)

# 加载所有实验结果
def load_results():
    results = {}
    
    # exp01: Baseline
    with open(results_dir / 'exp01_baseline_metrics.json') as f:
        results['Baseline'] = json.load(f)
    
    # exp02: Temperature Scaling
    with open(results_dir / 'exp02_overall_calibration.json') as f:
        results['Temperature'] = json.load(f)
    
    # exp03: Isotonic Regression
    with open(results_dir / 'exp03_overall_isotonic.json') as f:
        results['Isotonic'] = json.load(f)
    
    # exp04: Focal Loss
    with open(results_dir / 'exp04_focal_loss.json') as f:
        focal_data = json.load(f)
        results['Focal Loss'] = focal_data.get('Focal', focal_data.get('BCE', {}))
    
    # exp05: Label Smoothing
    with open(results_dir / 'exp05_label_smoothing.json') as f:
        ls_data = json.load(f)
        results['Label Smoothing'] = ls_data.get('LS_0.1', {})
    
    # exp07: Regularized Isotonic
    with open(results_dir / 'exp07_regularized_isotonic.json') as f:
        reg_data = json.load(f)
        results['Regularized Iso'] = reg_data.get('regularized', {}).get('0.01', {})
    
    # exp10: Grouped Isotonic
    with open(results_dir / 'exp10_grouped_isotonic.json') as f:
        grouped_data = json.load(f)
        results['Grouped Isotonic'] = grouped_data.get('grouped', {})
    
    # exp11: Two-Stage
    with open(results_dir / 'exp11_two_stage.json') as f:
        two_stage_data = json.load(f)
        results['Two-Stage'] = two_stage_data.get('two_stage', {})
    
    return results

# 图 1: 所有方法对比
def plot_comparison(results):
    methods = list(results.keys())
    aucs = [results[m].get('auc', 0) for m in methods]
    eces = [results[m].get('ece', 0) for m in methods]
    pcocs = [results[m].get('pcoc', 0) for m in methods]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # AUC
    colors_auc = ['#ff6b6b' if m == 'Grouped Isotonic' else '#4ecdc4' for m in methods]
    bars1 = axes[0].bar(methods, aucs, color=colors_auc, alpha=0.7)
    axes[0].set_title('AUC (排序能力)', fontsize=14)
    axes[0].set_ylabel('AUC')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=max(aucs), color='red', linestyle='--', label=f'Best: {max(aucs):.4f}')
    axes[0].legend()
    
    # ECE (越小越好)
    colors_ece = ['#95e1d3' if m == 'Grouped Isotonic' or m == 'Isotonic' else '#f38181' for m in methods]
    bars2 = axes[1].bar(methods, eces, color=colors_ece, alpha=0.7)
    axes[1].set_title('ECE (校准误差，越小越好)', fontsize=14)
    axes[1].set_ylabel('ECE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # PCOC (越接近 1 越好)
    colors_pcoc = ['#a8e6cf' if abs(pcoc - 1.0) < 0.01 else '#fd8b8b' for pcoc in pcocs]
    bars3 = axes[2].bar(methods, pcocs, color=colors_pcoc, alpha=0.7)
    axes[2].set_title('PCOC (越接近 1.0 越好)', fontsize=14)
    axes[2].set_ylabel('PCOC')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].axhline(y=1.0, color='green', linestyle='--', label='Ideal: 1.0')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存：all_methods_comparison.png")

# 图 2: Top business_type 分组校准效果
def plot_grouped_bt():
    with open(results_dir / 'exp10_grouped_isotonic.json') as f:
        data = json.load(f)
    
    bt_results = data.get('by_business_type', [])
    if not bt_results:
        print("⚠️ 无 BT 数据")
        return
    
    bts = [r['business_type'] for r in bt_results[:10]]
    samples_pct = [r['pct'] for r in bt_results[:10]]
    ece_before = [r.get('ece_before', 0) for r in bt_results[:10]]
    ece_after = [r.get('ece', 0) for r in bt_results[:10]]
    
    x = np.arange(len(bts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, ece_before, width, label='整体 Isotonic', alpha=0.7, color='#ff6b6b')
    bars2 = ax.bar(x + width/2, ece_after, width, label='Grouped Isotonic', alpha=0.7, color='#4ecdc4')
    
    ax.set_xlabel('Business Type', fontsize=12)
    ax.set_ylabel('ECE', fontsize=12)
    ax.set_title('Top 10 Business Type - 分组 vs 整体校准对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'BT {bt}' for bt in bts])
    ax.legend()
    
    # 添加样本占比注释
    for i, pct in enumerate(samples_pct):
        ax.text(i, max(ece_before+ECE_after)/2 + 0.001, f'{pct:.1f}%', 
               ha='center', va='bottom', fontsize=9, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'grouped_vs_overall_by_bt.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存：grouped_vs_overall_by_bt.png")

# 图 3: 校准曲线
def plot_calibration_curve():
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    
    df = pd.read_parquet(results_dir / 'exp01b_predictions.parquet')
    preds = df['pred'].values
    labels = df['label'].values
    
    # 分箱统计
    n_bins = 20
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline
    ax1 = axes[0]
    bin_centers = []
    bin_accuracies = []
    for i in range(n_bins):
        mask = (preds > bin_boundaries[i]) & (preds <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_accuracies.append(labels[mask].mean())
    
    ax1.plot(bin_centers, bin_accuracies, 'o-', label='Baseline', color='#ff6b6b', linewidth=2, markersize=8)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Actual Probability', fontsize=12)
    ax1.set_title('Baseline Calibration Curve', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Isotonic
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    calibrated = iso_reg.predict(preds)
    
    bin_centers_iso = []
    bin_accuracies_iso = []
    for i in range(n_bins):
        mask = (calibrated > bin_boundaries[i]) & (calibrated <= bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_centers_iso.append((bin_boundaries[i] + bin_boundaries[i+1]) / 2)
            bin_accuracies_iso.append(labels[mask].mean())
    
    ax2 = axes[1]
    ax2.plot(bin_centers_iso, bin_accuracies_iso, 'o-', label='Isotonic', color='#4ecdc4', linewidth=2, markersize=8)
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax2.set_xlabel('Calibrated Probability', fontsize=12)
    ax2.set_ylabel('Actual Probability', fontsize=12)
    ax2.set_title('Isotonic Regression Calibration Curve', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存：calibration_curves.png")

if __name__ == '__main__':
    print("="*60)
    print("生成 CTR/CVR 校准研究可视化图表")
    print("="*60)
    
    results = load_results()
    print(f"\n加载了 {len(results)} 个实验结果")
    
    plot_comparison(results)
    plot_calibration_curve()
    
    print(f"\n✅ 所有图表已保存到 {output_dir}")
