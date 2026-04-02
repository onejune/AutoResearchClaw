#!/usr/bin/env python3
"""
exp001 v3: AutoEmb 维度分配策略消融

对比不同的启发式维度分配规则：
- v3a: 激进压缩（更多低维）
- v3b: 保守压缩（更多高维）
- v3c: 基于频次的动态分配
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.loader import IVRDataLoader, collate_fn
from src.train.trainer import DynamicTrainer
from src.models.autoemb import AutoEmbWideDeep
from src.utils.hardware_monitor import HardwareMonitor


def suggest_dims_aggressive(feature_config):
    """激进压缩：尽量用低维"""
    dims = {}
    for feat, vocab_size in feature_config.items():
        if vocab_size > 1_000_000:
            dims[feat] = 16  # 超稀疏用更低维
        elif vocab_size > 100_000:
            dims[feat] = 32
        elif vocab_size > 10_000:
            dims[feat] = 32
        elif vocab_size > 1_000:
            dims[feat] = 16
        else:
            dims[feat] = 8  # 密集特征用最低维
    return dims


def suggest_dims_conservative(feature_config):
    """保守压缩：保留高维"""
    dims = {}
    for feat, vocab_size in feature_config.items():
        if vocab_size > 1_000_000:
            dims[feat] = 64
        elif vocab_size > 100_000:
            dims[feat] = 128
        elif vocab_size > 10_000:
            dims[feat] = 64
        elif vocab_size > 1_000:
            dims[feat] = 64
        else:
            dims[feat] = 32
    return dims


def suggest_dims_log_scale(feature_config):
    """对数尺度：dim = min(128, max(8, 4 * log2(vocab)))"""
    import math
    dims = {}
    for feat, vocab_size in feature_config.items():
        if vocab_size <= 1:
            dims[feat] = 8
        else:
            dim = int(4 * math.log2(vocab_size))
            dim = max(8, min(128, dim))
            # 向上取整到 8 的倍数
            dim = ((dim + 7) // 8) * 8
            dims[feat] = dim
    return dims


STRATEGIES = {
    "v3a_aggressive": suggest_dims_aggressive,
    "v3b_conservative": suggest_dims_conservative,
    "v3c_log_scale": suggest_dims_log_scale,
}


def load_shared_baseline():
    baseline_dir = Path("results/shared_baseline")
    if not baseline_dir.exists():
        return None, None, None
    
    with open(baseline_dir / "results.json") as f:
        results = json.load(f)
    with open(baseline_dir / "feature_config.json") as f:
        feature_config = json.load(f)
    
    bt_grouped = {}
    if (baseline_dir / "bt_grouped.json").exists():
        with open(baseline_dir / "bt_grouped.json") as f:
            bt_grouped = json.load(f)
    
    return results, feature_config, bt_grouped


def run_experiment(strategy_name, dim_func, device, data_loader, feature_config, config):
    print(f"\n{'='*60}")
    print(f"策略: {strategy_name}")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # 计算维度分配
    auto_dims = dim_func(feature_config)
    
    # 统计维度分布
    dim_dist = {}
    for d in auto_dims.values():
        dim_dist[d] = dim_dist.get(d, 0) + 1
    print(f"  维度分布: {dim_dist}")
    
    # 计算参数量
    total_params = sum(vocab * auto_dims[feat] for feat, vocab in feature_config.items())
    baseline_params = sum(vocab * 64 for vocab in feature_config.values())
    reduction = (1 - total_params / baseline_params) * 100
    print(f"  参数量减少: {reduction:.1f}%")
    
    model = AutoEmbWideDeep(
        feature_config=feature_config,
        dimension_config=auto_dims,
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    trainer = DynamicTrainer(model=model, lr=config["lr"], device=device)
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    
    results = trainer.evaluate(test_loader)
    training_time = time.time() - start_time
    
    return {
        "strategy": strategy_name,
        "dim_distribution": dim_dist,
        "param_reduction_pct": reduction,
        "auc": results["auc"],
        "pcoc": results["pcoc"],
        "logloss": results["logloss"],
        "training_time_sec": training_time,
        "bt_grouped": results.get("bt_grouped", {}),
    }


def main():
    print("=" * 60)
    print("exp001 v3: AutoEmb 维度策略消融实验")
    print("=" * 60)
    
    config = {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "seed": 42,
    }
    
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    
    baseline_results, feature_config, baseline_bt = load_shared_baseline()
    feature_config = {k: int(v) for k, v in feature_config.items()}
    
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    results_all = []
    for strategy_name, dim_func in STRATEGIES.items():
        result = run_experiment(strategy_name, dim_func, device, data_loader, feature_config, config)
        results_all.append(result)
    
    # 保存
    results_dir = Path("results/exp001_autoemb_v3")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "experiment": "exp001_autoemb_v3",
        "baseline": baseline_results,
        "results": results_all,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    print(f"{'策略':<20} {'AUC':<12} {'PCOC':<12} {'参数减少':<12} {'vs Baseline':<15}")
    print("-" * 80)
    
    if baseline_results:
        print(f"{'Baseline (64d)':<20} {baseline_results['auc']:<12.4f} {baseline_results['pcoc']:<12.4f} {'0%':<12} {'-':<15}")
    
    for r in results_all:
        delta = r['auc'] - baseline_results['auc'] if baseline_results else 0
        marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
        print(f"{r['strategy']:<20} {r['auc']:<12.4f} {r['pcoc']:<12.4f} {r['param_reduction_pct']:.1f}%{'':<6} {delta:+.4f} {marker}")
    
    print(f"\n✅ 完成！结果: {results_dir}/")


if __name__ == "__main__":
    main()
