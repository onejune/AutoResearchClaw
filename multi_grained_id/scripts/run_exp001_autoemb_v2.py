#!/usr/bin/env python3
"""
exp001: AutoEmb 自动维度分配实验 (v2 - 使用公共 Baseline)

改进点:
1. 复用 shared_baseline 结果，不重复训练 Baseline
2. 智能 GPU 选择，利用空闲卡
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
from src.models.autoemb import AutoEmbWideDeep, suggest_dimensions_heuristic, compare_parameter_efficiency
from src.utils.hardware_monitor import HardwareMonitor


def load_shared_baseline():
    """加载公共 Baseline 结果"""
    baseline_dir = Path("results/shared_baseline")
    
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"公共 Baseline 不存在，请先运行: python scripts/train_shared_baseline.py"
        )
    
    with open(baseline_dir / "results.json") as f:
        results = json.load(f)
    
    with open(baseline_dir / "feature_config.json") as f:
        feature_config = json.load(f)
    
    bt_grouped = {}
    bt_file = baseline_dir / "bt_grouped.json"
    if bt_file.exists():
        with open(bt_file) as f:
            bt_grouped = json.load(f)
    
    print(f"✅ 已加载公共 Baseline")
    print(f"   AUC: {results['auc']:.4f}")
    print(f"   PCOC: {results['pcoc']:.4f}")
    print(f"   训练时间: {results['training_time_sec']:.2f}s")
    
    return results, feature_config, bt_grouped


def main():
    print("=" * 60)
    print("exp001: AutoEmb 自动维度分配实验 (v2)")
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
    
    # 加载公共 Baseline
    print("\n[1/4] 加载公共 Baseline...")
    try:
        baseline_results, feature_config, baseline_bt = load_shared_baseline()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("正在训练公共 Baseline...")
        os.system("python scripts/train_shared_baseline.py")
        baseline_results, feature_config, baseline_bt = load_shared_baseline()
    
    # 智能选择设备（选择空闲的 GPU）
    print("\n[2/4] 选择训练设备...")
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    print(f"选中设备: {device} ({device_info.name})")
    
    # 计算参数效率
    print("\n[3/4] 计算参数效率...")
    
    # 转换 feature_config 为正确格式
    feature_config_int = {k: int(v) for k, v in feature_config.items()}
    
    auto_dims = suggest_dimensions_heuristic(feature_config_int)
    param_stats = compare_parameter_efficiency(feature_config_int, fixed_dim=64, auto_dims=auto_dims)
    
    print(f"\n固定 64 维方案:")
    print(f"  - Embedding 参数量：{param_stats['fixed_dim_scheme']['params_mb']:.2f} MB")
    
    print(f"\nAutoEmb 方案:")
    print(f"  - Embedding 参数量：{param_stats['auto_dim_scheme']['params_mb']:.2f} MB")
    print(f"  - 维度分布：{param_stats['auto_dim_scheme']['dim_distribution']}")
    
    print(f"\n节省:")
    print(f"  - 参数量减少：{param_stats['savings']['param_reduction']:.2f}%")
    print(f"  - 节省空间：{param_stats['savings']['mb_saved']:.2f} MB")
    
    # 保存参数统计
    results_dir = Path("results/exp001_autoemb")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "param_stats.json", "w") as f:
        json.dump(param_stats, f, indent=2)
    
    # 加载数据（只需要加载一次）
    print("\n[4/4] 训练 AutoEmb 模型...")
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
    
    # 训练 AutoEmb
    autoemb_model = AutoEmbWideDeep(
        feature_config=feature_config_int,
        dimension_config=auto_dims,
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    autoemb_trainer = DynamicTrainer(
        model=autoemb_model,
        lr=config["lr"],
        device=device,
    )
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        autoemb_trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    autoemb_results = autoemb_trainer.evaluate(test_loader)
    autoemb_time = time.time() - start_time
    
    print(f"\nAutoEmb 训练时间：{autoemb_time:.2f} 秒")
    print(f"AutoEmb AUC: {autoemb_results['auc']:.4f}")
    print(f"AutoEmb PCOC: {autoemb_results['pcoc']:.4f}")
    
    # 获取参数统计
    autoemb_param_stats = autoemb_model.get_parameter_stats()
    
    # 保存模型
    torch.save(autoemb_model.state_dict(), results_dir / "model_autoemb.pt")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    final_results = {
        "experiment": "exp001_autoemb",
        "config": config,
        "dimension_suggestions": auto_dims,
        "parameter_efficiency": param_stats,
        "baseline": {
            "auc": baseline_results["auc"],
            "pcoc": baseline_results["pcoc"],
            "logloss": baseline_results.get("logloss", 0),
            "training_time_sec": baseline_results["training_time_sec"],
            "bt_grouped": baseline_bt,
            "source": "shared_baseline (复用)",
        },
        "autoemb": {
            "auc": autoemb_results["auc"],
            "pcoc": autoemb_results["pcoc"],
            "logloss": autoemb_results["logloss"],
            "training_time_sec": autoemb_time,
            "bt_grouped": autoemb_results.get("bt_grouped", {}),
            "param_stats": autoemb_param_stats,
            "device": str(device),
        },
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印对比表
    print(f"\n{'指标':<15} {'Baseline':<15} {'AutoEmb':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {autoemb_results['auc']:<15.4f} {(autoemb_results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {autoemb_results['pcoc']:<15.4f} {(autoemb_results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    print(f"{'训练时间 (s)':<15} {baseline_results['training_time_sec']:<15.2f} {autoemb_time:<15.2f} {'(Baseline 复用)':>15}")
    print(f"{'Embedding 参数':<15} {param_stats['fixed_dim_scheme']['params_mb']:<15.2f}MB {param_stats['auto_dim_scheme']['params_mb']:<15.2f}MB {-param_stats['savings']['mb_saved']:>+15.2f}MB")
    
    # 分 BT 对比
    if baseline_bt and autoemb_results.get("bt_grouped"):
        print("\n分 Business Type 对比 (Top 10 by 样本量):")
        print("-" * 80)
        print(f"{'BT':<12} {'样本数':<12} {'Baseline AUC':<15} {'AutoEmb AUC':<15} {'Delta':<15}")
        print("-" * 80)
        
        bt_sorted = sorted(baseline_bt.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        
        for bt_name, bt_data in bt_sorted:
            baseline_auc = bt_data["auc"]
            autoemb_auc = autoemb_results.get("bt_grouped", {}).get(bt_name, {}).get("auc", 0)
            delta = autoemb_auc - baseline_auc
            marker = "🔺" if delta > 0.005 else ("🔻" if delta < -0.005 else "")
            print(f"{bt_name:<12} {bt_data['count']:<12,} {baseline_auc:<15.4f} {autoemb_auc:<15.4f} {delta:+15.4f} {marker}")
    
    print("\n" + "=" * 60)
    print("✅ 实验完成！结果已保存到 results/exp001_autoemb/")
    print("=" * 60)


if __name__ == "__main__":
    main()
