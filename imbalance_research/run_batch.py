#!/usr/bin/env python3
"""
并行批量实验运行器 - 充分利用多 GPU

用法:
    # 并行运行所有实验 (自动检测 GPU 数量)
    python run_batch.py
    
    # 指定并行数
    python run_batch.py --parallel 2
    
    # 调试模式
    python run_batch.py --debug
"""

import argparse
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).parent


def get_gpu_count() -> int:
    """获取可用 GPU 数量"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        return len(result.stdout.strip().split('\n'))
    except:
        return 0


def run_experiment(exp_name: str, debug: bool = False) -> Dict[str, Any]:
    """运行单个实验 (子进程)"""
    cmd = [sys.executable, 'run.py', '--exp', exp_name]
    if debug:
        cmd.append('--debug')
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200  # 2 小时超时
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # 解析结果
            output_lines = result.stdout.strip().split('\n')
            auc_line = [l for l in output_lines if '最佳 Val AUC' in l]
            auc = float(auc_line[0].split(':')[-1].strip()) if auc_line else 0.0
            
            return {
                'experiment': exp_name,
                'status': 'completed',
                'auc': auc,
                'elapsed_seconds': elapsed,
            }
        else:
            return {
                'experiment': exp_name,
                'status': 'failed',
                'error': result.stderr[-500:] if result.stderr else 'Unknown error',
                'elapsed_seconds': elapsed,
            }
    
    except subprocess.TimeoutExpired:
        return {
            'experiment': exp_name,
            'status': 'timeout',
            'elapsed_seconds': 7200,
        }
    except Exception as e:
        return {
            'experiment': exp_name,
            'status': 'error',
            'error': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description='并行批量实验')
    parser.add_argument('--parallel', type=int, default=None, help='并行数 (默认=GPU数)')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--experiments', type=str, nargs='+', help='指定实验列表')
    args = parser.parse_args()
    
    # 获取实验列表
    if args.experiments:
        experiments = args.experiments
    else:
        # 从 registry 获取所有实验
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.registry import EXPERIMENTS
        experiments = list(EXPERIMENTS.keys())
    
    # 确定并行数
    gpu_count = get_gpu_count()
    parallel = args.parallel or max(1, gpu_count)
    
    print("\n" + "=" * 70)
    print("🚀 并行批量实验运行器")
    print("=" * 70)
    print(f"   GPU 数量: {gpu_count}")
    print(f"   并行数: {parallel}")
    print(f"   实验数: {len(experiments)}")
    print(f"   调试模式: {'是' if args.debug else '否'}")
    print(f"   开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n📋 实验列表:")
    for i, exp in enumerate(experiments, 1):
        print(f"   {i:2d}. {exp}")
    print()
    
    # 并行执行
    results = []
    completed = 0
    failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(run_experiment, exp, args.debug): exp 
            for exp in experiments
        }
        
        for future in as_completed(futures):
            exp_name = futures[future]
            result = future.result()
            results.append(result)
            
            if result['status'] == 'completed':
                completed += 1
                print(f"✅ [{completed + failed}/{len(experiments)}] {exp_name}: "
                      f"AUC={result.get('auc', 0):.4f} ({result['elapsed_seconds']:.0f}s)")
            else:
                failed += 1
                print(f"❌ [{completed + failed}/{len(experiments)}] {exp_name}: "
                      f"{result['status']} - {result.get('error', '')[:50]}")
    
    total_time = time.time() - start_time
    
    # 汇总
    print("\n" + "=" * 70)
    print("📊 实验汇总")
    print("=" * 70)
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   完成: {completed}/{len(experiments)}")
    print(f"   失败: {failed}/{len(experiments)}")
    
    # 按 AUC 排序
    completed_results = [r for r in results if r['status'] == 'completed']
    if completed_results:
        completed_results.sort(key=lambda x: x.get('auc', 0), reverse=True)
        
        print("\n🏆 AUC 排行榜:")
        for i, r in enumerate(completed_results[:10], 1):
            print(f"   {i:2d}. {r['experiment']:30s} AUC={r['auc']:.4f}")
    
    # 保存结果
    summary_path = PROJECT_ROOT / 'results' / 'batch_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'start_time': datetime.now().isoformat(),
        'total_experiments': len(experiments),
        'completed': completed,
        'failed': failed,
        'total_seconds': total_time,
        'debug_mode': args.debug,
        'results': results,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 汇总已保存: {summary_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
