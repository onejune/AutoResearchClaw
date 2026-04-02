#!/usr/bin/env python3
"""
exp006: FiBiNET + AutoFIS - 特征重要性 + 交互选择实验

使用 SENet 学习特征重要性，AutoFIS 自动选择有效特征交互。
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
from src.models.fibinet import FiBiNETWideDeep
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
    print("exp006: FiBiNET + AutoFIS 实验")
    print("=" * 60)
    
    config = {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "embedding_size": 64,
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "bilinear_type": "field_all",
        "use_senet": True,
        "use_autofis": True,
        "senet_reduction": 4,
        "autofis_temperature": 1.0,
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
    
    # 为了减少计算量，只选择部分重要特征
    # 选择 vocab size 较大的特征（通常是 ID 类特征）
    sorted_features = sorted(feature_config_int.items(), key=lambda x: x[1], reverse=True)
    top_features = dict(sorted_features[:30])  # 选择 top 30 特征
    
    print(f"选择 Top 30 特征（共 {len(feature_config_int)} 个）")
    print(f"特征交互数量: {30 * 29 // 2} = 435 对")
    
    # 创建 FiBiNET 模型
    print("\n[4/4] 训练 FiBiNET 模型...")
    model = FiBiNETWideDeep(
        feature_config=top_features,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
        bilinear_type=config["bilinear_type"],
        use_senet=config["use_senet"],
        use_autofis=config["use_autofis"],
        senet_reduction=config["senet_reduction"],
        autofis_temperature=config["autofis_temperature"],
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
    
    # 获取统计信息
    param_stats = model.get_parameter_stats()
    selected_interactions = model.get_selected_interactions()[:20]  # Top 20 交互
    
    # 保存结果
    results_dir = Path("results/exp006_fibinet")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / "model_fibinet.pt")
    
    final_results = {
        "experiment": "exp006_fibinet",
        "config": config,
        "selected_features": list(top_features.keys()),
        "device": str(device),
        "baseline": baseline_results,
        "baseline_bt": baseline_bt,
        "fibinet": {
            "auc": results["auc"],
            "pcoc": results["pcoc"],
            "logloss": results["logloss"],
            "training_time_sec": training_time,
            "bt_grouped": results.get("bt_grouped", {}),
        },
        "parameter_stats": param_stats,
        "top_interactions": [
            {"feat1": f1, "feat2": f2, "prob": p}
            for f1, f2, p in selected_interactions
        ],
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'Baseline':<15} {'FiBiNET':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {results['auc']:<15.4f} {(results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {results['pcoc']:<15.4f} {(results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    
    print(f"\n参数统计:")
    print(f"  总参数量: {param_stats['total_params']:,}")
    print(f"  特征数量: {param_stats['num_features']}")
    print(f"  交互数量: {param_stats['num_interactions']}")
    
    print(f"\nTop 10 特征交互 (by AutoFIS):")
    for i, (f1, f2, prob) in enumerate(selected_interactions[:10], 1):
        print(f"  {i}. {f1} × {f2}: {prob:.4f}")
    
    # 分 BT 对比
    if baseline_bt and results.get("bt_grouped"):
        print("\n分 BT 对比 (Top 10):")
        print("-" * 70)
        bt_sorted = sorted(baseline_bt.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        for bt_name, bt_data in bt_sorted:
            baseline_auc = bt_data["auc"]
            fib_auc = results.get("bt_grouped", {}).get(bt_name, {}).get("auc", 0)
            if fib_auc > 0:
                delta = fib_auc - baseline_auc
                marker = "🔺" if delta > 0.005 else ("🔻" if delta < -0.005 else "")
                print(f"  {bt_name}: Baseline={baseline_auc:.4f}, FiBiNET={fib_auc:.4f}, Delta={delta:+.4f} {marker}")
    
    print("\n" + "=" * 60)
    print(f"✅ 实验完成！结果已保存到 {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
