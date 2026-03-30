#!/usr/bin/env python3
"""
运行所有对比学习实验
1. cl_weight 调参（bt_contrastive）
2. augment_contrastive 实验
"""
import os
import sys
import subprocess
import json
from datetime import datetime

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
RESULTS_DIR = f'{PROJECT_DIR}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_RESULTS = []

def run_experiment(model, cl_weight=0.1):
    cmd = [
        'python', f'{PROJECT_DIR}/train.py',
        '--model', model,
        '--cl_weight', str(cl_weight),
        '--epochs', '1'
    ]
    print(f"\n{'='*60}")
    print(f"Running {model} with cl_weight={cl_weight}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
    
    # 解析结果
    auc = None
    for line in result.stdout.split('\n'):
        if 'AUC:' in line and 'Val' not in line and 'BT' not in line:
            try:
                auc = float(line.split('AUC:')[1].split()[0])
            except:
                pass
    return {'model': model, 'cl_weight': cl_weight, 'auc': auc}

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*60)
    print("IVR SSL CVR 完整实验")
    print("="*60)
    
    # 1. Baseline（已有结果，但重跑确认）
    print("\n[1/6] Baseline")
    r = run_experiment('baseline')
    ALL_RESULTS.append(r)
    print(f">>> Baseline AUC: {r['auc']}")
    
    # 2. BT_Contrastive cl_weight 调参
    for cw in [0.05, 0.1, 0.2, 0.5]:
        print(f"\n[{2 + [0.05, 0.1, 0.2, 0.5].index(cw)}/6] BT_Contrastive cl_weight={cw}")
        r = run_experiment('bt_contrastive', cw)
        ALL_RESULTS.append(r)
        print(f">>> BT_Contrastive (cw={cw}) AUC: {r['auc']}")
    
    # 3. Augment Contrastive
    print("\n[6/6] Augment_Contrastive")
    r = run_experiment('augment_contrastive', 0.1)
    ALL_RESULTS.append(r)
    print(f">>> Augment_Contrastive AUC: {r['auc']}")
    
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
    result_file = f'{RESULTS_DIR}/all_experiments_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print(f"\n结果已保存到: {result_file}")
    
    # 找最佳
    valid_results = [r for r in ALL_RESULTS if r['auc'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['auc'])
        print(f"\n🏆 最佳: {best['model']} (cl_weight={best['cl_weight']}) AUC={best['auc']:.4f}")
