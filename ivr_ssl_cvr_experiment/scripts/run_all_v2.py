#!/usr/bin/env python3
"""
运行所有对比学习实验 v2 - 使用 data_v2 数据集，支持用户对比学习
"""
import os
import sys
import subprocess
import json
from datetime import datetime

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
RESULTS_DIR = f'{PROJECT_DIR}/results'
DATA_DIR = f'{PROJECT_DIR}/data_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_RESULTS = []

def run_experiment(model, cl_weight=0.1):
    cmd = [
        'python', f'{PROJECT_DIR}/train_v2.py',
        '--model', model,
        '--cl_weight', str(cl_weight),
        '--epochs', '1',
        '--data_dir', DATA_DIR
    ]
    print(f"\n{'='*60}")
    print(f"Running {model} with cl_weight={cl_weight}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
    
    # 解析结果
    auc = None
    for line in result.stdout.split('\n'):
        if 'Final AUC:' in line:
            try:
                auc = float(line.split('Final AUC:')[1].strip())
            except:
                pass
    return {'model': model, 'cl_weight': cl_weight, 'auc': auc}

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("IVR SSL CVR 完整实验 v2（含用户对比学习）")
    print("="*60)
    
    experiments = [
        ('baseline', 0.1),
        ('bt_contrastive', 0.05),
        ('bt_contrastive', 0.1),
        ('bt_contrastive', 0.2),
        ('user_contrastive', 0.1),
        ('augment_contrastive', 0.1),
    ]
    
    for i, (model, cw) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {model} cl_weight={cw}")
        r = run_experiment(model, cw)
        ALL_RESULTS.append(r)
        print(f">>> {model} (cw={cw}) AUC: {r['auc']}")
    
    # 汇总
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    print(f"{'Model':<25} {'cl_weight':<12} {'AUC':<10}")
    print("-"*47)
    for r in ALL_RESULTS:
        auc_str = f"{r['auc']:.4f}" if r['auc'] else 'N/A'
        print(f"{r['model']:<25} {r['cl_weight']:<12} {auc_str:<10}")
    
    # 保存
    result_file = f'{RESULTS_DIR}/all_experiments_v2_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n结果已保存到: {result_file}")
    
    # 找最佳
    valid_results = [r for r in ALL_RESULTS if r['auc'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['auc'])
        baseline_auc = next((r['auc'] for r in ALL_RESULTS if r['model'] == 'baseline'), None)
        print(f"\n🏆 最佳: {best['model']} (cl_weight={best['cl_weight']}) AUC={best['auc']:.4f}")
        if baseline_auc:
            gain = (best['auc'] - baseline_auc) * 1000
            print(f"   相比 Baseline 提升: {gain:.2f} 千分点")
