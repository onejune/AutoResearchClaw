#!/usr/bin/env python3
"""
训练公共 Baseline 模型

所有实验共用这个 Baseline，避免重复训练。
输出：
- results/shared_baseline/model.pt (模型权重)
- results/shared_baseline/results.json (评估结果)
- results/shared_baseline/bt_grouped.json (分 BT 结果)
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
from src.models.widedeep import WideDeep
from src.utils.hardware_monitor import HardwareMonitor


def main():
    print("=" * 60)
    print("训练公共 Baseline 模型")
    print("=" * 60)
    
    # 配置
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
    
    # 设置随机种子
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # 智能选择设备
    print("\n[1/4] 选择训练设备...")
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    print(f"选中设备: {device} ({device_info.name})")
    
    # 数据加载
    print("\n[2/4] 加载数据...")
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
    
    feature_config = data_loader.get_feature_config()
    
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"特征数量：{len(feature_config)}")
    
    # 创建模型
    print("\n[3/4] 训练 Baseline 模型...")
    model = WideDeep(
        feature_config=feature_config,
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
    
    print(f"\n训练时间：{training_time:.2f} 秒")
    print(f"AUC: {results['auc']:.4f}")
    print(f"PCOC: {results['pcoc']:.4f}")
    print(f"LogLoss: {results['logloss']:.4f}")
    
    # 保存结果
    print("\n[4/4] 保存 Baseline...")
    results_dir = Path("results/shared_baseline")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), results_dir / "model.pt")
    
    # 保存配置
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 保存结果
    final_results = {
        "auc": results["auc"],
        "pcoc": results["pcoc"],
        "logloss": results["logloss"],
        "training_time_sec": training_time,
        "device": str(device),
        "device_name": device_info.name,
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    # 保存分 BT 结果
    if "bt_grouped" in results:
        with open(results_dir / "bt_grouped.json", "w") as f:
            json.dump(results["bt_grouped"], f, indent=2)
    
    # 保存特征配置（供后续实验使用）
    with open(results_dir / "feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)
    
    print(f"\n✅ Baseline 已保存到 {results_dir}/")
    print("后续实验可直接加载，无需重复训练")


if __name__ == "__main__":
    main()
