#!/usr/bin/env python3
"""
exp001: AutoEmb 自动维度分配实验

对比方案:
- Baseline: 所有特征固定 64 维
- AutoEmb: 按启发式规则自动分配维度
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import IVRDataLoader, collate_fn
from src.train.trainer import DynamicTrainer as Trainer
from src.models.widedeep import WideDeep
from src.models.autoemb import AutoEmbWideDeep, suggest_dimensions_heuristic, compare_parameter_efficiency


def setup_experiment():
    """实验配置"""
    return {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "fixed_dim": 64,
        "embedding_size": 64,  # baseline 用这个
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "seed": 42,
    }


def main():
    print("=" * 60)
    print("exp001: AutoEmb 自动维度分配实验")
    print("=" * 60)
    
    # 设置配置
    config = setup_experiment()
    
    # 设置随机种子
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # 数据加载
    print("\n[1/4] 加载数据...")
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
    
    print(f"\n训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"特征数量：{len(feature_config)}")
    
    # 计算参数效率对比
    print("\n[2/4] 计算参数效率...")
    auto_dims = suggest_dimensions_heuristic(feature_config)
    param_stats = compare_parameter_efficiency(feature_config, fixed_dim=64, auto_dims=auto_dims)
    
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
    
    # ========== Baseline: 固定 64 维 ==========
    print("\n" + "=" * 60)
    print("[3/4] 训练 Baseline (固定 64 维)...")
    print("=" * 60)
    
    baseline_model = WideDeep(
        feature_config=feature_config,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    baseline_trainer = Trainer(
        model=baseline_model,
        lr=config["lr"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        baseline_trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    baseline_results = baseline_trainer.evaluate(test_loader)
    baseline_time = time.time() - start_time
    
    print(f"\nBaseline 训练时间：{baseline_time:.2f} 秒")
    print(f"Baseline AUC: {baseline_results['auc']:.4f}")
    print(f"Baseline PCOC: {baseline_results['pcoc']:.4f}")
    
    # 保存 baseline 模型和结果
    torch.save(baseline_model.state_dict(), results_dir / "model_baseline.pt")
    
    # ========== AutoEmb: 自动分配维度 ==========
    print("\n" + "=" * 60)
    print("[4/4] 训练 AutoEmb (自动维度分配)...")
    print("=" * 60)
    
    autoemb_model = AutoEmbWideDeep(
        feature_config=feature_config,
        dimension_config=auto_dims,
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    autoemb_trainer = Trainer(
        model=autoemb_model,
        lr=config["lr"],
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    print(f"\nAutoEmb 参数统计:")
    print(f"  - 总参数量：{autoemb_param_stats['total_params'] / 1e6:.2f} M")
    print(f"  - Embedding 参数量：{autoemb_param_stats['embedding_params'] / 1e6:.2f} M")
    print(f"  - 平均 embedding 维度：{autoemb_param_stats['avg_embedding_dim']:.2f}")
    
    # 保存 autoemb 模型和结果
    torch.save(autoemb_model.state_dict(), results_dir / "model_autoemb.pt")
    
    # ========== 汇总结果 ==========
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
            "logloss": baseline_results["logloss"],
            "training_time_sec": baseline_time,
            "bt_grouped": baseline_results.get("bt_grouped", {}),
        },
        "autoemb": {
            "auc": autoemb_results["auc"],
            "pcoc": autoemb_results["pcoc"],
            "logloss": autoemb_results["logloss"],
            "training_time_sec": autoemb_time,
            "bt_grouped": autoemb_results.get("bt_grouped", {}),
            "param_stats": autoemb_param_stats,
        },
    }
    
    # 保存完整结果
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印对比表
    print(f"\n{'指标':<15} {'Baseline':<15} {'AutoEmb':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {autoemb_results['auc']:<15.4f} {(autoemb_results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {autoemb_results['pcoc']:<15.4f} {(autoemb_results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    print(f"{'LogLoss':<15} {baseline_results['logloss']:<15.4f} {autoemb_results['logloss']:<15.4f} {(autoemb_results['logloss'] - baseline_results['logloss']):>+15.4f}")
    print(f"{'训练时间 (s)':<15} {baseline_time:<15.2f} {autoemb_time:<15.2f} {(autoemb_time - baseline_time):>+15.2f}")
    print(f"{'Embedding 参数':<15} {param_stats['fixed_dim_scheme']['params_mb']:<15.2f}MB {param_stats['auto_dim_scheme']['params_mb']:<15.2f}MB {-param_stats['savings']['mb_saved']:>+15.2f}MB")
    
    # 分 BT 对比（Top 10）
    print("\n分 Business Type 对比 (Top 10 by 样本量):")
    print("-" * 80)
    print(f"{'BT':<12} {'样本数':<12} {'Baseline AUC':<15} {'AutoEmb AUC':<15} {'Delta':<15}")
    print("-" * 80)
    
    bt_grouped = baseline_results.get("bt_grouped", {})
    if bt_grouped:
        # 按样本量排序
        bt_sorted = sorted(bt_grouped.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        
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


def setup_experiment():
    """实验配置"""
    return {
        "label_col": "ctcvr_label",
        "batch_size": 512,
        "lr": 5e-5,
        "epochs": 1,
        "fixed_dim": 64,
        "embedding_size": 64,  # baseline 用这个
        "dnn_hidden_units": [1024, 512, 256, 128],
        "dropout": 0.3,
        "seed": 42,
    }


def main():
    print("=" * 60)
    print("exp001: AutoEmb 自动维度分配实验")
    print("=" * 60)
    
    # 设置配置
    config = setup_experiment()
    
    # 设置随机种子
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # 数据加载
    print("\n[1/4] 加载数据...")
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
        collate_fn=lambda x: x  # 简单占位，实际使用 dataset 的 collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x
    )
    
    feature_config = data_loader.get_feature_config()
    
    print(f"\n训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"特征数量：{len(feature_config)}")
    
    # 计算参数效率对比
    print("\n[2/4] 计算参数效率...")
    auto_dims = suggest_dimensions_heuristic(feature_config)
    param_stats = compare_parameter_efficiency(feature_config, fixed_dim=64, auto_dims=auto_dims)
    
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
    
    # ========== Baseline: 固定 64 维 ==========
    print("\n" + "=" * 60)
    print("[3/4] 训练 Baseline (固定 64 维)...")
    print("=" * 60)
    
    baseline_model = WideDeep(
        feature_config=feature_config,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    baseline_trainer = Trainer(
        model=baseline_model,
        learning_rate=config["lr"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        baseline_trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    baseline_results = baseline_trainer.evaluate(test_loader)
    baseline_time = time.time() - start_time
    
    print(f"\nBaseline 训练时间：{baseline_time:.2f} 秒")
    print(f"Baseline AUC: {baseline_results['auc']:.4f}")
    print(f"Baseline PCOC: {baseline_results['pcoc']:.4f}")
    
    # 保存 baseline 模型和结果
    torch.save(baseline_model.state_dict(), results_dir / "model_baseline.pt")
    
    # ========== AutoEmb: 自动分配维度 ==========
    print("\n" + "=" * 60)
    print("[4/4] 训练 AutoEmb (自动维度分配)...")
    print("=" * 60)
    
    autoemb_model = AutoEmbWideDeep(
        feature_config=feature_config,
        dimension_config=auto_dims,
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
    )
    
    autoemb_trainer = Trainer(
        model=autoemb_model,
        learning_rate=config["lr"],
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    print(f"\nAutoEmb 参数统计:")
    print(f"  - 总参数量：{autoemb_param_stats['total_params'] / 1e6:.2f} M")
    print(f"  - Embedding 参数量：{autoemb_param_stats['embedding_params'] / 1e6:.2f} M")
    print(f"  - 平均 embedding 维度：{autoemb_param_stats['avg_embedding_dim']:.2f}")
    
    # 保存 autoemb 模型和结果
    torch.save(autoemb_model.state_dict(), results_dir / "model_autoemb.pt")
    
    # ========== 汇总结果 ==========
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
            "logloss": baseline_results["logloss"],
            "training_time_sec": baseline_time,
            "bt_grouped": baseline_results["bt_grouped"],
        },
        "autoemb": {
            "auc": autoemb_results["auc"],
            "pcoc": autoemb_results["pcoc"],
            "logloss": autoemb_results["logloss"],
            "training_time_sec": autoemb_time,
            "bt_grouped": autoemb_results["bt_grouped"],
            "param_stats": autoemb_param_stats,
        },
    }
    
    # 保存完整结果
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印对比表
    print(f"\n{'指标':<15} {'Baseline':<15} {'AutoEmb':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {autoemb_results['auc']:<15.4f} {(autoemb_results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {autoemb_results['pcoc']:<15.4f} {(autoemb_results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    print(f"{'LogLoss':<15} {baseline_results['logloss']:<15.4f} {autoemb_results['logloss']:<15.4f} {(autoemb_results['logloss'] - baseline_results['logloss']):>+15.4f}")
    print(f"{'训练时间 (s)':<15} {baseline_time:<15.2f} {autoemb_time:<15.2f} {(autoemb_time - baseline_time):>+15.2f}")
    print(f"{'Embedding 参数':<15} {param_stats['fixed_dim_scheme']['params_mb']:<15.2f}MB {param_stats['auto_dim_scheme']['params_mb']:<15.2f}MB {-param_stats['savings']['mb_saved']:>+15.2f}MB")
    
    # 分 BT 对比（Top 10）
    print("\n分 Business Type 对比 (Top 10 by 样本量):")
    print("-" * 80)
    print(f"{'BT':<12} {'样本数':<12} {'Baseline AUC':<15} {'AutoEmb AUC':<15} {'Delta':<15}")
    print("-" * 80)
    
    # 按样本量排序
    bt_sorted = sorted(baseline_results["bt_grouped"].items(), 
                       key=lambda x: x[1]["count"], 
                       reverse=True)[:10]
    
    for bt_name, bt_data in bt_sorted:
        baseline_auc = bt_data["auc"]
        autoemb_auc = autoemb_results["bt_grouped"][bt_name]["auc"]
        delta = autoemb_auc - baseline_auc
        marker = "🔺" if delta > 0.005 else ("🔻" if delta < -0.005 else "")
        print(f"{bt_name:<12} {bt_data['count']:<12,} {baseline_auc:<15.4f} {autoemb_auc:<15.4f} {delta:+15.4f} {marker}")
    
    print("\n" + "=" * 60)
    print("✅ 实验完成！结果已保存到 results/exp001_autoemb/")
    print("=" * 60)


if __name__ == "__main__":
    main()
