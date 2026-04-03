#!/usr/bin/env python3
"""
exp004: MetaEmb - 元学习冷启动实验

使用元学习为冷启动 ID 生成初始 embedding。
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
from src.models.metaemb import MetaEmbWideDeep
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
    print("exp004: MetaEmb 元学习冷启动实验")
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
    
    # 定义元学习特征对：{ID 特征: [side information 特征]}
    # 这些 side features 用于为冷启动 ID 生成 embedding
    meta_features = {
        "campaignid": ["campaignsetid", "offerid", "business_type"],
        "demand_pkgname": ["business_type", "offerid"],
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
    
    # 过滤有效的 meta features
    valid_meta_features = {}
    for id_feat, side_feats in meta_features.items():
        if id_feat in feature_config_int:
            valid_sides = [sf for sf in side_feats if sf in feature_config_int]
            if valid_sides:
                valid_meta_features[id_feat] = valid_sides
    
    print(f"有效的 Meta 特征配置: {valid_meta_features}")
    
    # 创建 MetaEmb 模型
    print("\n[4/4] 训练 MetaEmb 模型...")
    model = MetaEmbWideDeep(
        feature_config=feature_config_int,
        meta_features=valid_meta_features,
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
    
    param_stats = model.get_parameter_stats()
    
    # 保存结果
    results_dir = Path("results/exp004_metaemb")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / "model_metaemb.pt")
    
    final_results = {
        "experiment": "exp004_metaemb",
        "config": config,
        "meta_features": valid_meta_features,
        "device": str(device),
        "baseline": baseline_results,
        "baseline_bt": baseline_bt,
        "metaemb": {
            "auc": results["auc"],
            "pcoc": results["pcoc"],
            "logloss": results["logloss"],
            "training_time_sec": training_time,
            "bt_grouped": results.get("bt_grouped", {}),
        },
        "parameter_stats": param_stats,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'指标':<15} {'Baseline':<15} {'MetaEmb':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {results['auc']:<15.4f} {(results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {results['pcoc']:<15.4f} {(results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    
    print(f"\n参数统计:")
    print(f"  总参数量: {param_stats['total_params']:,}")
    print(f"  Embedding 参数量: {param_stats['embedding_params']:,}")
    print(f"  Meta Generator 参数量: {param_stats['meta_generator_params']:,}")
    
    # 分 BT 对比
    if baseline_bt and results.get("bt_grouped"):
        print("\n分 BT 对比 (Top 10):")
        print("-" * 70)
        bt_sorted = sorted(baseline_bt.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        for bt_name, bt_data in bt_sorted:
            baseline_auc = bt_data["auc"]
            meta_auc = results.get("bt_grouped", {}).get(bt_name, {}).get("auc", 0)
            if meta_auc > 0:
                delta = meta_auc - baseline_auc
                marker = "🔺" if delta > 0.005 else ("🔻" if delta < -0.005 else "")
                print(f"  {bt_name}: Baseline={baseline_auc:.4f}, MetaEmb={meta_auc:.4f}, Delta={delta:+.4f} {marker}")
    
    print("\n" + "=" * 60)
    print(f"✅ 实验完成！结果已保存到 {results_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
