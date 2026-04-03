#!/usr/bin/env python3
"""
exp007: Combined - 组合最优方法

基于前面实验的最优结果，组合多种技术：
- Hierarchical Embedding（PCOC 改善）
- AutoEmb 维度分配（参数压缩）
- SENet 特征重要性（可选）

目标：在保持 AUC 的同时，改善 PCOC 并减少参数量
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.loader import IVRDataLoader, collate_fn
from src.train.trainer import DynamicTrainer
from src.models.hierarchical import HierarchicalEmbedding
from src.utils.hardware_monitor import HardwareMonitor


class CombinedWideDeep(nn.Module):
    """组合模型：Hierarchical + AutoEmb + SENet"""
    
    def __init__(
        self,
        feature_config,
        hierarchical_pairs,
        dimension_config,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3,
        use_senet=True,
        senet_reduction=4
    ):
        super().__init__()
        
        self.feature_config = feature_config
        self.hierarchical_pairs = hierarchical_pairs
        self.dimension_config = dimension_config
        self.use_senet = use_senet
        
        # 识别层次化特征和普通特征
        self.hier_fine_features = set(hierarchical_pairs.keys())
        self.hier_coarse_features = set(hierarchical_pairs.values())
        self.plain_features = [f for f in feature_config.keys() 
                               if f not in self.hier_fine_features and f not in self.hier_coarse_features]
        
        # 层次化 Embedding
        self.hierarchical_embeddings = nn.ModuleDict()
        for fine, coarse in hierarchical_pairs.items():
            if fine in feature_config and coarse in feature_config:
                emb_dim = dimension_config.get(fine, 64)
                self.hierarchical_embeddings[fine] = HierarchicalEmbedding(
                    fine_vocab_size=feature_config[fine],
                    coarse_vocab_size=feature_config[coarse],
                    embedding_dim=emb_dim
                )
        
        # 普通 Embedding（使用 AutoEmb 维度）
        self.plain_embeddings = nn.ModuleDict()
        for feat in self.plain_features:
            if feat in feature_config:
                emb_dim = dimension_config.get(feat, 64)
                self.plain_embeddings[feat] = nn.Embedding(
                    feature_config[feat] + 1, emb_dim, padding_idx=0
                )
        
        # 计算总 embedding 维度
        total_dim = 0
        for fine in self.hierarchical_embeddings:
            total_dim += dimension_config.get(fine, 64)
        for feat in self.plain_features:
            total_dim += dimension_config.get(feat, 64)
        
        # SENet（可选）
        num_features = len(self.hierarchical_embeddings) + len(self.plain_features)
        if use_senet and num_features > 1:
            reduced_dim = max(1, num_features // senet_reduction)
            self.senet = nn.Sequential(
                nn.Linear(num_features, reduced_dim),
                nn.ReLU(),
                nn.Linear(reduced_dim, num_features),
                nn.Sigmoid()
            )
        else:
            self.senet = None
        
        # DNN
        layers = []
        prev_dim = total_dim
        for hidden_dim in dnn_hidden_units:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.dnn = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # 初始化
        for emb in self.plain_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, features):
        embeddings = []
        gate_values = {}
        
        # 层次化 embedding
        for fine, hier_emb in self.hierarchical_embeddings.items():
            coarse = self.hierarchical_pairs[fine]
            if fine in features and coarse in features:
                emb, gate = hier_emb(features[fine], features[coarse])
                embeddings.append(emb)
                gate_values[fine] = gate.mean().item()
        
        # 普通 embedding
        for feat, emb_layer in self.plain_embeddings.items():
            if feat in features:
                embeddings.append(emb_layer(features[feat]))
        
        if not embeddings:
            raise ValueError("No valid features")
        
        # SENet 重加权（可选）
        if self.senet is not None and len(embeddings) > 1:
            # 计算每个 embedding 的均值作为 squeeze
            squeezed = torch.stack([e.mean(dim=-1) for e in embeddings], dim=-1)
            weights = self.senet(squeezed)
            embeddings = [e * w.unsqueeze(-1) for e, w in zip(embeddings, weights.unbind(-1))]
        
        # 拼接
        x = torch.cat(embeddings, dim=-1)
        
        # DNN
        x = self.dnn(x)
        logits = self.output_layer(x).squeeze(-1)
        
        return logits
    
    def get_parameter_stats(self):
        total = sum(p.numel() for p in self.parameters())
        hier_params = sum(sum(p.numel() for p in e.parameters()) 
                         for e in self.hierarchical_embeddings.values())
        plain_params = sum(sum(p.numel() for p in e.parameters())
                          for e in self.plain_embeddings.values())
        return {
            "total_params": total,
            "hierarchical_params": hier_params,
            "plain_embedding_params": plain_params,
        }


def suggest_dims_balanced(feature_config):
    """平衡策略：适中的维度分配"""
    dims = {}
    for feat, vocab_size in feature_config.items():
        if vocab_size > 1_000_000:
            dims[feat] = 32
        elif vocab_size > 100_000:
            dims[feat] = 64
        elif vocab_size > 10_000:
            dims[feat] = 48
        elif vocab_size > 1_000:
            dims[feat] = 32
        else:
            dims[feat] = 16
    return dims


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


def main():
    print("=" * 60)
    print("exp007: Combined 组合最优方法实验")
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
    
    # 最优层次化配置（来自 exp003 v1）
    hierarchical_pairs = {
        "campaignid": "campaignsetid",
        "demand_pkgname": "business_type",
    }
    
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    monitor = HardwareMonitor()
    device, device_info = monitor.select_device(min_memory_mb=4096)
    
    baseline_results, feature_config, baseline_bt = load_shared_baseline()
    feature_config = {k: int(v) for k, v in feature_config.items()}
    
    # AutoEmb 维度分配
    dimension_config = suggest_dims_balanced(feature_config)
    
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # 创建组合模型
    model = CombinedWideDeep(
        feature_config=feature_config,
        hierarchical_pairs=hierarchical_pairs,
        dimension_config=dimension_config,
        dnn_hidden_units=config["dnn_hidden_units"],
        dropout=config["dropout"],
        use_senet=True,
    )
    
    trainer = DynamicTrainer(model=model, lr=config["lr"], device=device)
    
    start_time = time.time()
    for epoch in range(1, config["epochs"] + 1):
        trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config["epochs"])
    
    results = trainer.evaluate(test_loader)
    training_time = time.time() - start_time
    
    param_stats = model.get_parameter_stats()
    
    # 保存
    results_dir = Path("results/exp007_combined")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), results_dir / "model_combined.pt")
    
    final_results = {
        "experiment": "exp007_combined",
        "config": config,
        "hierarchical_pairs": hierarchical_pairs,
        "baseline": baseline_results,
        "combined": {
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
    
    # 打印
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(f"{'指标':<15} {'Baseline':<15} {'Combined':<15} {'Delta':<15}")
    print("-" * 60)
    print(f"{'AUC':<15} {baseline_results['auc']:<15.4f} {results['auc']:<15.4f} {(results['auc'] - baseline_results['auc']):>+15.4f}")
    print(f"{'PCOC':<15} {baseline_results['pcoc']:<15.4f} {results['pcoc']:<15.4f} {(results['pcoc'] - baseline_results['pcoc']):>+15.4f}")
    
    print(f"\n参数统计: {param_stats}")
    print(f"\n✅ 完成！结果: {results_dir}/")


if __name__ == "__main__":
    main()
