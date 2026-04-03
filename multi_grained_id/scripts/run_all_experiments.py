#!/usr/bin/env python3
"""
自动运行所有实验的调度脚本

功能：
1. 检测 GPU 空闲状态
2. 自动分配实验到空闲 GPU
3. 监控实验进度
4. 生成汇总报告
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime


# 实验队列（按优先级排序）
EXPERIMENTS = [
    # 基础实验
    ("exp001_autoemb_v2", "run_exp001_autoemb_v2.py", "AutoEmb 自动维度"),
    ("exp002_dds", "run_exp002_dds.py", "DDS 可微分搜索"),
    ("exp004_metaemb", "run_exp004_metaemb.py", "MetaEmb 元学习"),
    ("exp005_contrastive", "run_exp005_contrastive.py", "Contrastive 对比学习"),
    ("exp006_fibinet", "run_exp006_fibinet.py", "FiBiNET 特征交互"),
    
    # 消融实验
    ("exp003_hierarchical_v3", "run_exp003_hierarchical_v3.py", "Hierarchical 层级消融"),
    ("exp001_autoemb_v3", "run_exp001_autoemb_v3.py", "AutoEmb 策略消融"),
    
    # 组合实验
    ("exp007_combined", "run_exp007_combined.py", "Combined 组合最优"),
]


def get_gpu_status():
    """获取 GPU 状态"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 4:
                    gpus.append({
                        'index': int(parts[0].strip()),
                        'utilization': float(parts[1].strip()),
                        'memory_used': float(parts[2].strip()),
                        'memory_total': float(parts[3].strip()),
                    })
            return gpus
    except Exception as e:
        print(f"获取 GPU 状态失败: {e}")
    return []


def get_free_gpu(gpus, min_memory_mb=4096, max_util=50):
    """获取空闲 GPU"""
    for gpu in gpus:
        free_memory = gpu['memory_total'] - gpu['memory_used']
        if gpu['utilization'] < max_util and free_memory > min_memory_mb:
            return gpu['index']
    return None


def check_experiment_done(exp_name):
    """检查实验是否已完成"""
    results_dir = Path(f"results/{exp_name}")
    results_file = results_dir / "results.json"
    return results_file.exists()


def run_experiment(script_name, gpu_id):
    """在指定 GPU 上运行实验"""
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/{script_name}"
    log_file = f"/tmp/{script_name.replace('.py', '')}.log"
    
    full_cmd = f"cd /mnt/workspace/open_research/autoresearch/multi_grained_id && {cmd} > {log_file} 2>&1 &"
    
    subprocess.run(full_cmd, shell=True)
    return log_file


def get_running_experiments():
    """获取正在运行的实验"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, timeout=5
        )
        running = []
        for line in result.stdout.split('\n'):
            if 'python scripts/run_exp' in line and 'grep' not in line:
                for exp_name, script, _ in EXPERIMENTS:
                    if script in line:
                        running.append(exp_name)
        return running
    except Exception:
        return []


def main():
    print("=" * 70)
    print("Multi-Grained ID 实验自动调度器")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 检查 Baseline
    baseline_dir = Path("results/shared_baseline")
    if not baseline_dir.exists():
        print("\n❌ 公共 Baseline 不存在，请先运行: python scripts/train_shared_baseline.py")
        return
    
    print("\n✅ 公共 Baseline 已就绪")
    
    # 实验状态
    print("\n实验状态:")
    print("-" * 70)
    
    pending = []
    completed = []
    
    for exp_name, script, desc in EXPERIMENTS:
        if check_experiment_done(exp_name):
            completed.append((exp_name, desc))
            print(f"  ✅ {exp_name}: {desc} (已完成)")
        else:
            pending.append((exp_name, script, desc))
            print(f"  ⏳ {exp_name}: {desc} (待运行)")
    
    if not pending:
        print("\n🎉 所有实验已完成！")
        generate_summary()
        return
    
    print(f"\n待运行: {len(pending)} 个实验")
    
    # 调度循环
    while pending:
        gpus = get_gpu_status()
        running = get_running_experiments()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] GPU 状态:")
        for gpu in gpus:
            status = "🔴 繁忙" if gpu['utilization'] > 50 else "🟢 空闲"
            print(f"  GPU {gpu['index']}: {gpu['utilization']:.0f}% 利用率, "
                  f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB {status}")
        
        print(f"  正在运行: {running if running else '无'}")
        
        # 找空闲 GPU
        free_gpu = get_free_gpu(gpus)
        
        if free_gpu is not None and pending:
            exp_name, script, desc = pending.pop(0)
            
            # 再次检查是否已完成（可能被其他进程完成）
            if check_experiment_done(exp_name):
                print(f"  ⏭️ {exp_name} 已完成，跳过")
                continue
            
            print(f"\n🚀 启动 {exp_name} (GPU {free_gpu}): {desc}")
            run_experiment(script, free_gpu)
            
            # 等待启动
            time.sleep(10)
        else:
            # 等待 GPU 空闲
            print(f"  等待 GPU 空闲...")
            time.sleep(30)
        
        # 检查已完成的实验
        new_pending = []
        for exp_name, script, desc in pending:
            if check_experiment_done(exp_name):
                completed.append((exp_name, desc))
                print(f"  ✅ {exp_name} 完成")
            else:
                new_pending.append((exp_name, script, desc))
        pending = new_pending
    
    print("\n🎉 所有实验已完成！")
    generate_summary()


def generate_summary():
    """生成实验汇总报告"""
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    # 加载 Baseline
    baseline_dir = Path("results/shared_baseline")
    with open(baseline_dir / "results.json") as f:
        baseline = json.load(f)
    
    print(f"\n{'实验':<25} {'AUC':<12} {'PCOC':<12} {'vs Baseline':<15}")
    print("-" * 70)
    print(f"{'Baseline':<25} {baseline['auc']:<12.4f} {baseline['pcoc']:<12.4f} {'-':<15}")
    
    results_data = []
    
    for exp_name, _, desc in EXPERIMENTS:
        results_file = Path(f"results/{exp_name}/results.json")
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            
            # 提取 AUC 和 PCOC
            if 'combined' in data:
                auc = data['combined']['auc']
                pcoc = data['combined']['pcoc']
            elif 'dds' in data:
                auc = data['dds']['auc']
                pcoc = data['dds']['pcoc']
            elif 'metaemb' in data:
                auc = data['metaemb']['auc']
                pcoc = data['metaemb']['pcoc']
            elif 'contrastive' in data:
                auc = data['contrastive']['auc']
                pcoc = data['contrastive']['pcoc']
            elif 'fibinet' in data:
                auc = data['fibinet']['auc']
                pcoc = data['fibinet']['pcoc']
            elif 'autoemb' in data:
                auc = data['autoemb']['auc']
                pcoc = data['autoemb']['pcoc']
            elif 'hierarchical_results' in data and data['hierarchical_results']:
                auc = data['hierarchical_results'][0]['auc']
                pcoc = data['hierarchical_results'][0]['pcoc']
            elif 'results' in data and data['results']:
                auc = data['results'][0]['auc']
                pcoc = data['results'][0]['pcoc']
            else:
                continue
            
            delta = auc - baseline['auc']
            marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
            
            print(f"{exp_name:<25} {auc:<12.4f} {pcoc:<12.4f} {delta:+.4f} {marker}")
            results_data.append({
                "experiment": exp_name,
                "description": desc,
                "auc": auc,
                "pcoc": pcoc,
                "delta_auc": delta
            })
    
    # 保存汇总
    summary_file = Path("results/experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "baseline": baseline,
            "experiments": results_data
        }, f, indent=2)
    
    print(f"\n汇总已保存: {summary_file}")


if __name__ == "__main__":
    main()
