#!/usr/bin/env python3
"""
exp002: Data Distribution Search (DDS) - 自动维度搜索

使用可微分搜索方法，为每个特征自动选择最优 embedding 维度。
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
from src.models.dds import DDSWideDeep
from src.utils.hardware_monitor import HardwareMonitor


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


def main():
    print("=" * 60)
    print("exp002: DDS 自动维度搜索实验")
    print("=" * 60)
    
    config = {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "candidate_dims": [8, 16, 32, 64, 128],
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "temperature": 1.0,
        "seed": 42,
    }
    
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # 智能选择设备
    print("\n[1/4] 选择训练设备...")
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    print(f"选中设备: {device} ({device_info.name})")
    
    # 加载公共 Baseline
    print("\n[2/4] 加载 Baseline...")
    baseline_results, feature_config, baseline_bt = load_shared_baseline()
    if baseline_results:
        print(f"✅ 已加载公共 Baseline: AUC={baseline_results['auc']:.4f}")
    else:
        print("⚠️ 公共 Baseline 不存在，请先运行 train_shared_baseline.py")
        return
    
    # 加载数据
    print("\n[3/4] 加载数据...")
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
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
    
    feature_config_int = {k: int(v) for k, v in feature_config.items()}
    
    # 创建 DDS 模型
    print("\n[4/4] 训练 DDS 模型...")
    model = DDSWideDeep(
        feature_config=feature_config_int,
        candidate_dims=config["candidate_dims"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
        temperature=config["temperature"],
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
    
    # 获取维度选择统计
    dim_stats = model.get_dimension_stats()
    param_stats = model.get_parameter_stats()
    
    # 保存结果
    results_dir = Path("results/exp002_dds")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / "model_dds.pt")
    
    final_results = {
        "experiment": "exp002_dds",
        "config": config,
        "device": str(device),
        "baseline": baseline_results,
        "baseline_bt": baseline_bt,
        "dds": {
            "auc": results["auc"],
            "pcoc": results["pcoc"],
            "logloss": results["logloss"],
            "training_time_sec": training_time,
            "bt_grouped": results.get("bt_grouped", {}),
        },
        "dimension_stats": dim_stats,
        "parameter_stats": param_stats,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'Baseline':<15} {'DDS':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {results['auc']:<15.4f} {(results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {results['pcoc']:<15.4f} {(results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    
    # 维度分布统计
    print("\n维度选择分布 (Top 10 by vocab size):")
    sorted_feats = sorted(dim_stats.items(), key=lambda x: feature_config_int.get(x[0], 0), reverse=True)[:10]
    for feat, stats in sorted_feats:
        print(f"  {feat}: selected={stats['selected_dim']}d, dist={stats['distribution']}")
    
    print(f"\n平均有效维度: {param_stats['avg_effective_dim']:.1f}")
    
    print("\n" + "=" * 60)
    print(f"✅ 实验完成！结果已保存到 {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
