#!/usr/bin/env python3
"""
统一实验入口 - 简洁、可靠、不会跑丢

用法:
    # 单个实验
    python run.py --exp baseline_bce
    
    # 批量实验
    python run.py --batch experiments/batch.yaml
    
    # 快速测试
    python run.py --exp baseline_bce --debug
    
    # 指定 GPU
    python run.py --exp baseline_bce --gpu 0
"""

import argparse
import os
import sys
import json
import yaml
import time
import torch
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.experiment import Experiment
from core.registry import EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser(description="Focal Loss 类别不均衡研究")
    
    # 实验选择
    parser.add_argument('--exp', type=str, help='实验名称 (见 experiments/registry.py)')
    parser.add_argument('--batch', type=str, help='批量实验配置文件')
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    
    # 运行模式
    parser.add_argument('--debug', action='store_true', help='调试模式 (小数据, 2 epochs)')
    parser.add_argument('--gpu', type=int, default=-1, help='指定 GPU (-1=自动)')
    parser.add_argument('--resume', type=str, help='从 checkpoint 恢复')
    
    # 覆盖参数
    parser.add_argument('--epochs', type=int, help='覆盖 epochs')
    parser.add_argument('--batch-size', type=int, help='覆盖 batch_size')
    parser.add_argument('--lr', type=float, help='覆盖 learning_rate')
    
    return parser.parse_args()


def list_experiments():
    """列出所有已注册实验"""
    print("\n📋 可用实验列表:")
    print("=" * 60)
    
    for name, config in EXPERIMENTS.items():
        desc = config.get('description', '无描述')
        loss = config.get('loss', {}).get('type', 'unknown')
        model = config.get('model', {}).get('type', 'unknown')
        print(f"  {name:30s} | {loss:12s} | {model}")
    
    print("=" * 60)
    print(f"共 {len(EXPERIMENTS)} 个实验\n")


def run_single(exp_name: str, args) -> Dict[str, Any]:
    """运行单个实验"""
    if exp_name not in EXPERIMENTS:
        print(f"❌ 未知实验: {exp_name}")
        print(f"💡 使用 --list 查看可用实验")
        sys.exit(1)
    
    config = EXPERIMENTS[exp_name].copy()
    
    # 应用命令行覆盖
    if args.debug:
        config['training'] = config.get('training', {})
        config['training']['epochs'] = 2
        config['training']['batch_size'] = 64
        config['data'] = config.get('data', {})
        config['data']['sample_ratio'] = 0.01  # 1% 数据
    
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.gpu >= 0:
        config.setdefault('gpu', {})['device'] = args.gpu
    
    # 创建并运行实验
    exp = Experiment(name=exp_name, config=config)
    
    if args.resume:
        exp.load_checkpoint(args.resume)
    
    return exp.run()


def run_batch(batch_file: str, args) -> List[Dict[str, Any]]:
    """批量运行实验"""
    with open(batch_file) as f:
        batch_config = yaml.safe_load(f)
    
    experiments = batch_config.get('experiments', [])
    results = []
    
    print(f"\n🚀 批量实验: {len(experiments)} 个")
    print("=" * 60)
    
    for i, exp_name in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp_name}")
        print("-" * 40)
        
        try:
            result = run_single(exp_name, args)
            result['status'] = 'completed'
            results.append(result)
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            results.append({
                'experiment': exp_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存汇总
    summary_path = PROJECT_ROOT / 'results' / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 批量实验完成，汇总保存至: {summary_path}")
    return results


def main():
    args = parse_args()
    
    if args.list:
        list_experiments()
        return
    
    if not args.exp and not args.batch:
        print("❌ 请指定 --exp 或 --batch")
        print("💡 使用 --list 查看可用实验")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("👑 Focal Loss 类别不均衡研究")
    print("=" * 60)
    
    if args.batch:
        run_batch(args.batch, args)
    else:
        run_single(args.exp, args)


if __name__ == "__main__":
    main()
