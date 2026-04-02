#!/usr/bin/env python3
"""
exp003 v2: Hierarchical Embeddings - 调整层级顺序

对比不同的层次化特征组合：
- v1: campaignid→campaignsetid, offerid→demand_pkgname, demand_pkgname→business_type
- v2: 更深的链式结构 + 更多配对

实验目标：找到最优的层次化特征组合
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


# 不同的层次化配置
HIERARCHICAL_CONFIGS = {
    "v1_original": {
        "campaignid": "campaignsetid",
        "offerid": "demand_pkgname",
        "demand_pkgname": "business_type",
    },
    "v2_deep_chain": {
        # 更深的广告层级链
        "campaignid": "campaignsetid",
        "campaignsetid": "offerid",
        "offerid": "demand_pkgname",
        "demand_pkgname": "business_type",
    },
    "v3_device_hierarchy": {
        # 设备相关层级
        "model": "make",  # 机型 → 品牌
        "city": "region",  # 城市 → 区域（如果有）
        "campaignid": "campaignsetid",
        "demand_pkgname": "business_type",
    },
    "v4_minimal": {
        # 最小配置：只保留最重要的
        "demand_pkgname": "business_type",
    },
}


def load_shared_baseline():
    """加载公共 Baseline 结果"""
    baseline_dir = Path("results/shared_baseline")
    
    if not baseline_dir.exists():
        return None, None, None
    
    with open(baseline_dir / "results.json") as f:
        results = json.load(f)
    
    with open(baseline_dir / "feature_config.json") as f:
        feature_config = json.load(f)
    
    bt_grouped = {}
    bt_file = baseline_dir / "bt_grouped.json"
    if bt_file.exists():
        with open(bt_file) as f:
            bt_grouped = json.load(f)
    
    return results, feature_config, bt_grouped


def run_hierarchical_experiment(config_name, hierarchical_pairs, device, data_loader, config):
    """运行单个层次化实验"""
    print(f"\n{'='*60}")
    print(f"运行配置: {config_name}")
    print(f"层次化对: {hierarchical_pairs}")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    feature_config = data_loader.get_feature_config()
    
    # 过滤掉不存在的特征对
    valid_pairs = {}
    for fine, coarse in hierarchical_pairs.items():
        if fine in feature_config and coarse in feature_config:
            valid_pairs[fine] = coarse
        else:
            print(f"  ⚠️ 跳过不存在的特征对: {fine} → {coarse}")
    
    if not valid_pairs:
        print(f"  ❌ 没有有效的层次化特征对，跳过")
        return None
    
    print(f"  有效层次化对: {valid_pairs}")
    
    # 创建模型
    model = HierarchicalWideDeep(
        feature_config=feature_config,
        hierarchical_pairs=valid_pairs,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    trainer = DynamicTrainer(
        model=model,
        lr=config["lr"],
        device=device,
    )
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    
    results = trainer.evaluate(test_loader)
    training_time = time.time() - start_time
    
    # 获取门控统计
    gate_stats = {}
    if hasattr(model, 'get_gate_statistics'):
        gate_stats = model.get_gate_statistics()
    
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
    print("exp003 v2: Hierarchical Embeddings - 层级顺序消融实验")
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
    
    # 智能选择设备（会选择空闲的 GPU）
    print("\n[1/3] 选择训练设备...")
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    print(f"选中设备: {device} ({device_info.name})")
    
    # 尝试加载公共 Baseline
    print("\n[2/3] 加载 Baseline...")
    baseline_results, _, baseline_bt = load_shared_baseline()
    if baseline_results:
        print(f"✅ 已加载公共 Baseline: AUC={baseline_results['auc']:.4f}")
    else:
        print("⚠️ 公共 Baseline 不存在，将只对比不同层次化配置")
    
    # 加载数据
    print("\n[3/3] 加载数据...")
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    # 只跑 v2_deep_chain（老板要求的层级顺序调整）
    results_all = []
    
    config_name = "v2_deep_chain"
    hierarchical_pairs = HIERARCHICAL_CONFIGS[config_name]
    
    result = run_hierarchical_experiment(
        config_name, hierarchical_pairs, device, data_loader, config
    )
    
    if result:
        results_all.append(result)
    
    # 保存结果
    results_dir = Path("results/exp003_hierarchical_v2")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "experiment": "exp003_hierarchical_v2",
        "config": config,
        "device": str(device),
        "device_name": device_info.name,
        "baseline": baseline_results,
        "baseline_bt": baseline_bt,
        "hierarchical_results": results_all,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    
    print(f"\n{'配置':<20} {'AUC':<12} {'PCOC':<12} {'时间(s)':<12}")
    print("-" * 60)
    
    if baseline_results:
        print(f"{'Baseline':<20} {baseline_results['auc']:<12.4f} {baseline_results['pcoc']:<12.4f} {baseline_results.get('training_time_sec', 0):<12.1f}")
    
    for r in results_all:
        delta_auc = r['auc'] - baseline_results['auc'] if baseline_results else 0
        marker = "🔺" if delta_auc > 0.001 else ("🔻" if delta_auc < -0.001 else "")
        print(f"{r['config_name']:<20} {r['auc']:<12.4f} {r['pcoc']:<12.4f} {r['training_time_sec']:<12.1f} {marker}")
    
    # 分 BT 对比
    if results_all and baseline_bt:
        print("\n分 BT 对比 (vs Baseline):")
        print("-" * 80)
        print(f"{'BT':<12} {'样本数':<12} {'Baseline AUC':<15} {'v2_deep AUC':<15} {'Delta':<15}")
        print("-" * 80)
        
        bt_sorted = sorted(baseline_bt.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        
        for bt_name, bt_data in bt_sorted:
            baseline_auc = bt_data["auc"]
            v2_auc = results_all[0].get("bt_grouped", {}).get(bt_name, {}).get("auc", 0)
            if v2_auc > 0:
                delta = v2_auc - baseline_auc
                marker = "🔺" if delta > 0.005 else ("🔻" if delta < -0.005 else "")
                print(f"{bt_name:<12} {bt_data['count']:<12,} {baseline_auc:<15.4f} {v2_auc:<15.4f} {delta:+15.4f} {marker}")
    
    print("\n" + "=" * 80)
    print(f"✅ 实验完成！结果已保存到 {results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
