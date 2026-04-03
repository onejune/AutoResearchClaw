#!/usr/bin/env python3
"""
exp008: DeepFM 模型对比实验

DeepFM: FM 组件（二阶交互）+ Deep 组件（高阶交互）
与 WideDeep Baseline 对比
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
from src.models.deepfm import DeepFM
from src.utils.hardware_monitor import HardwareMonitor


def load_shared_baseline():
    """加载公共 Baseline 结果"""
    baseline_dir = Path(__file__).parent.parent / "results" / "shared_baseline"
    
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
    print("exp008: DeepFM 模型对比实验")
    print("=" * 60)
    
    config = {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "embedding_dim": 64,
        "hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "seed": 42,
    }
    
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # 智能选择设备
    print("\n[1/5] 选择训练设备...")
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    print(f"选中设备: {device} ({device_info.name})")
    
    # 加载公共 Baseline
    print("\n[2/5] 加载 Baseline...")
    baseline_results, feature_config, baseline_bt = load_shared_baseline()
    if baseline_results:
        print(f"✅ 已加载公共 Baseline: AUC={baseline_results['auc']:.4f}")
    else:
        print("⚠️ 公共 Baseline 不存在，请先运行 train_shared_baseline.py")
        return
    
    # 加载数据
    print("\n[3/5] 加载数据...")
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
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print("\n[4/5] 创建 DeepFM 模型...")
    model = DeepFM(
        field_dims=feature_config_int,
        embedding_dim=config["embedding_dim"],
        hidden_units=config["hidden_units"],
        dropout=config["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 训练
    print("\n[5/5] 训练模型...")
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
    
    # 保存结果
    print("\n[6/6] 保存结果...")
    result_dir = Path(__file__).parent.parent / "results" / "exp008_deepfm"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "experiment": "exp008_deepfm",
        "config": config,
        "device": str(device),
        "device_name": device_info.name,
        "baseline": baseline_results,
        "auc": results["auc"],
        "pcoc": results.get("pcoc"),
        "logloss": results.get("logloss"),
        "training_time_sec": training_time,
    }
    
    with open(result_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("实验完成!")
    print(f"{'='*60}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"PCOC: {results['pcoc']:.4f}" if results['pcoc'] else "")
    print(f"训练时间: {training_time:.1f} 秒")
    print(f"结果保存到: {result_dir}")


if __name__ == "__main__":
    main()
