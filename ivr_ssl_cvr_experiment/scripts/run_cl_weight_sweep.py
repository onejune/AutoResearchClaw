#!/usr/bin/env python3
"""
cl_weight 调参实验
"""
import os
import sys
import subprocess
import json
from datetime import datetime

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
sys.path.insert(0, PROJECT_DIR)

CL_WEIGHTS = [0.05, 0.1, 0.2, 0.5]
RESULTS = []

def run_experiment(cl_weight):
    cmd = [
        'python', f'{PROJECT_DIR}/train.py',
        '--model', 'bt_contrastive',
        '--cl_weight', str(cl_weight),
        '--epochs', '1'
    ]
    print(f"\n{'='*60}")
    print(f"Running BT_Contrastive with cl_weight={cl_weight}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    
    # 解析结果
    for line in result.stdout.split('\n'):
        if 'AUC:' in line and 'Val' not in line and 'BT' not in line:
            try:
                auc = float(line.split('AUC:')[1].split()[0])
                return {'cl_weight': cl_weight, 'auc': auc}
            except:
                pass
    return {'cl_weight': cl_weight, 'auc': None}

if __name__ == '__main__':
    print("="*60)
    print("CL_WEIGHT 调参实验")
    print("="*60)
    
    for cw in CL_WEIGHTS:
        result = run_experiment(cw)
        RESULTS.append(result)
        print(f"\n>>> cl_weight={cw}: AUC={result['auc']}")
    
    print("\n" + "="*60)
    print("最终结果汇总")
    print("="*60)
    print(f"{'cl_weight':<12} {'AUC':<10}")
    print("-"*22)
    for r in RESULTS:
        print(f"{r['cl_weight']:<12} {r['auc'] if r['auc'] else 'N/A':<10}")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'{PROJECT_DIR}/results/cl_weight_sweep_{timestamp}.json'
    with open(result_file, 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n结果已保存到: {result_file}")
