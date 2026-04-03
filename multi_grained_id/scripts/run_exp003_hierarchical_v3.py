#!/usr/bin/env python3
"""
exp003 v3: Hierarchical Embeddings - 最优层级组合消融

基于 v1/v2 结果，进一步探索最优的层次化组合：
- v1: 3 对（AUC=0.8454, PCOC=1.001）✅ 最优
- v2: 4 对 deep chain（AUC=0.8452, PCOC=1.016）略差

v3 消融：
- v3a: 只保留最有效的 1 对
- v3b: 只保留最有效的 2 对
- v3c: 加入设备层级（model→make）
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
from src.models.hierarchical import HierarchicalWideDeep
from src.utils.hardware_monitor import HardwareMonitor


CONFIGS = {
    "v3a_single_best": {
        # 只保留 demand_pkgname → business_type（最粗粒度）
        "demand_pkgname": "business_type",
    },
    "v3b_two_pairs": {
        # 保留效果最好的 2 对
        "campaignid": "campaignsetid",
        "demand_pkgname": "business_type",
    },
    "v3c_with_device": {
        # 加入设备层级
        "campaignid": "campaignsetid",
        "demand_pkgname": "business_type",
        "model": "make",  # 机型 → 品牌
    },
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


def run_experiment(config_name, hierarchical_pairs, device, data_loader, config):
    print(f"\n{'='*60}")
    print(f"配置: {config_name}")
    print(f"层次对: {hierarchical_pairs}")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                              shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    feature_config = data_loader.get_feature_config()
    
    # 过滤有效特征对
    valid_pairs = {k: v for k, v in hierarchical_pairs.items() 
                   if k in feature_config and v in feature_config}
    
    if not valid_pairs:
        print(f"  ❌ 无有效特征对")
        return None
    
    print(f"  有效特征对: {valid_pairs}")
    
    model = HierarchicalWideDeep(
        feature_config=feature_config,
        hierarchical_pairs=valid_pairs,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    trainer = DynamicTrainer(model=model, lr=config["lr"], device=device)
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    
    results = trainer.evaluate(test_loader)
    training_time = time.time() - start_time
    
    gate_stats = model.get_gate_statistics() if hasattr(model, 'get_gate_statistics') else {}
    
    return {
        "config_name": config_name,
        "hierarchical_pairs": valid_pairs,
        "auc": results["auc"],
        "pcoc": results["pcoc"],
        "logloss": results["logloss"],
        "training_time_sec": training_time,
        "bt_grouped": results.get("bt_grouped", {}),
        "gate_stats": gate_stats,
    }


def main():
    print("=" * 60)
    print("exp003 v3: Hierarchical 层级组合消融实验")
    print("=" * 60)
    
    config = {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "embedding_size": 64,
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "seed": 42,
    }
    
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    
    baseline_results, _, baseline_bt = load_shared_baseline()
    
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    results_all = []
    for config_name, pairs in CONFIGS.items():
        result = run_experiment(config_name, pairs, device, data_loader, config)
        if result:
            results_all.append(result)
    
    # 保存
    results_dir = Path("results/exp003_hierarchical_v3")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "experiment": "exp003_hierarchical_v3",
        "baseline": baseline_results,
        "results": results_all,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'配置':<20} {'AUC':<12} {'PCOC':<12} {'vs Baseline':<15}")
    print("-" * 70)
    
    if baseline_results:
        print(f"{'Baseline':<20} {baseline_results['auc']:<12.4f} {baseline_results['pcoc']:<12.4f} {'-':<15}")
    
    for r in results_all:
        delta = r['auc'] - baseline_results['auc'] if baseline_results else 0
        marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
        print(f"{r['config_name']:<20} {r['auc']:<12.4f} {r['pcoc']:<12.4f} {delta:+.4f} {marker}")
    
    print(f"\n✅ 完成！结果: {results_dir}/")


if __name__ == "__main__":
    main()
