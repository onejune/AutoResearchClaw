#!/usr/bin/env python3
"""
exp003 v5: Hierarchical Embeddings - 融合策略消融

探索不同的细粒度-粗粒度融合方式：
- v5a: 加权平均（当前方法）
- v5b: 拼接 + 投影
- v5c: 残差连接
- v5d: 双塔融合（分别过 MLP 再合并）
- v5e: 多头融合（多个门控头投票）
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.loader import IVRDataLoader, collate_fn
from src.train.trainer import DynamicTrainer
from src.utils.hardware_monitor import HardwareMonitor


class WeightedAvgHierarchicalEmbedding(nn.Module):
    """v5a: 加权平均（baseline）"""
    
    def __init__(self, fine_vocab, coarse_vocab, dim=64):
        super().__init__()
        self.fine_emb = nn.Embedding(fine_vocab + 1, dim, padding_idx=0)
        self.coarse_emb = nn.Embedding(coarse_vocab + 1, dim, padding_idx=0)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.fine_emb.weight)
        nn.init.xavier_uniform_(self.coarse_emb.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine = self.fine_emb(fine_ids)
        coarse = self.coarse_emb(coarse_ids)
        g = self.gate(torch.cat([fine, coarse], dim=-1))
        return g * fine + (1 - g) * coarse, g


class ConcatProjectHierarchicalEmbedding(nn.Module):
    """v5b: 拼接 + 投影"""
    
    def __init__(self, fine_vocab, coarse_vocab, dim=64):
        super().__init__()
        self.fine_emb = nn.Embedding(fine_vocab + 1, dim, padding_idx=0)
        self.coarse_emb = nn.Embedding(coarse_vocab + 1, dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        nn.init.xavier_uniform_(self.fine_emb.weight)
        nn.init.xavier_uniform_(self.coarse_emb.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine = self.fine_emb(fine_ids)
        coarse = self.coarse_emb(coarse_ids)
        combined = torch.cat([fine, coarse], dim=-1)
        return self.proj(combined), torch.tensor(0.5)  # 无门控


class ResidualHierarchicalEmbedding(nn.Module):
    """v5c: 残差连接"""
    
    def __init__(self, fine_vocab, coarse_vocab, dim=64):
        super().__init__()
        self.fine_emb = nn.Embedding(fine_vocab + 1, dim, padding_idx=0)
        self.coarse_emb = nn.Embedding(coarse_vocab + 1, dim, padding_idx=0)
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1), nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.fine_emb.weight)
        nn.init.xavier_uniform_(self.coarse_emb.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine = self.fine_emb(fine_ids)
        coarse = self.coarse_emb(coarse_ids)
        
        # 残差：fine + transform(coarse)
        residual = fine + self.transform(coarse)
        
        # 门控决定残差比例
        g = self.gate(torch.cat([fine, coarse], dim=-1))
        return g * residual + (1 - g) * coarse, g


class TwoTowerHierarchicalEmbedding(nn.Module):
    """v5d: 双塔融合"""
    
    def __init__(self, fine_vocab, coarse_vocab, dim=64):
        super().__init__()
        self.fine_emb = nn.Embedding(fine_vocab + 1, dim, padding_idx=0)
        self.coarse_emb = nn.Embedding(coarse_vocab + 1, dim, padding_idx=0)
        
        # 双塔 MLP
        self.fine_tower = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim // 2)
        )
        self.coarse_tower = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim // 2)
        )
        
        # 融合层
        self.fusion = nn.Linear(dim, dim)
        
        nn.init.xavier_uniform_(self.fine_emb.weight)
        nn.init.xavier_uniform_(self.coarse_emb.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine = self.fine_emb(fine_ids)
        coarse = self.coarse_emb(coarse_ids)
        
        fine_out = self.fine_tower(fine)
        coarse_out = self.coarse_tower(coarse)
        
        combined = torch.cat([fine_out, coarse_out], dim=-1)
        return self.fusion(combined), torch.tensor(0.5)


class MultiHeadHierarchicalEmbedding(nn.Module):
    """v5e: 多头融合（4 个门控头投票）"""
    
    def __init__(self, fine_vocab, coarse_vocab, dim=64, num_heads=4):
        super().__init__()
        self.fine_emb = nn.Embedding(fine_vocab + 1, dim, padding_idx=0)
        self.coarse_emb = nn.Embedding(coarse_vocab + 1, dim, padding_idx=0)
        self.num_heads = num_heads
        
        # 多个门控头
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
            )
            for _ in range(num_heads)
        ])
        
        # 头权重
        self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
        nn.init.xavier_uniform_(self.fine_emb.weight)
        nn.init.xavier_uniform_(self.coarse_emb.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine = self.fine_emb(fine_ids)
        coarse = self.coarse_emb(coarse_ids)
        combined = torch.cat([fine, coarse], dim=-1)
        
        # 多头门控
        gate_outputs = [gate(combined) for gate in self.gates]
        gates = torch.cat(gate_outputs, dim=-1)  # (batch, num_heads)
        
        # 加权平均
        weights = F.softmax(self.head_weights, dim=0)
        final_gate = (gates * weights).sum(dim=-1, keepdim=True)
        
        return final_gate * fine + (1 - final_gate) * coarse, final_gate


class HierarchicalWideDeepV5(nn.Module):
    """支持多种融合策略的 Hierarchical WideDeep"""
    
    def __init__(self, feature_config, hierarchical_pairs, fusion_type="weighted_avg",
                 embedding_size=64, dnn_hidden_units=[1024, 512, 256, 128], dropout=0.3):
        super().__init__()
        
        self.feature_config = feature_config
        self.hierarchical_pairs = hierarchical_pairs
        self.fusion_type = fusion_type
        
        fusion_classes = {
            "weighted_avg": WeightedAvgHierarchicalEmbedding,
            "concat_proj": ConcatProjectHierarchicalEmbedding,
            "residual": ResidualHierarchicalEmbedding,
            "two_tower": TwoTowerHierarchicalEmbedding,
            "multi_head": MultiHeadHierarchicalEmbedding,
        }
        FusionClass = fusion_classes.get(fusion_type, WeightedAvgHierarchicalEmbedding)
        
        self.hier_fine = set(hierarchical_pairs.keys())
        self.hier_coarse = set(hierarchical_pairs.values())
        self.plain_features = [f for f in feature_config if f not in self.hier_fine and f not in self.hier_coarse]
        
        self.hierarchical_embeddings = nn.ModuleDict()
        for fine, coarse in hierarchical_pairs.items():
            if fine in feature_config and coarse in feature_config:
                self.hierarchical_embeddings[fine] = FusionClass(
                    feature_config[fine], feature_config[coarse], embedding_size
                )
        
        self.plain_embeddings = nn.ModuleDict()
        for feat in self.plain_features:
            self.plain_embeddings[feat] = nn.Embedding(
                feature_config[feat] + 1, embedding_size, padding_idx=0
            )
        
        total_dim = (len(self.hierarchical_embeddings) + len(self.plain_features)) * embedding_size
        
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
        
        for emb in self.plain_embeddings.values():
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, features):
        embeddings = []
        
        for fine, hier_emb in self.hierarchical_embeddings.items():
            coarse = self.hierarchical_pairs[fine]
            if fine in features and coarse in features:
                emb, _ = hier_emb(features[fine], features[coarse])
                embeddings.append(emb)
        
        for feat, emb_layer in self.plain_embeddings.items():
            if feat in features:
                embeddings.append(emb_layer(features[feat]))
        
        x = torch.cat(embeddings, dim=-1)
        x = self.dnn(x)
        return self.output_layer(x).squeeze(-1)


FUSION_CONFIGS = {
    "v5a_weighted_avg": "weighted_avg",
    "v5b_concat_proj": "concat_proj",
    "v5c_residual": "residual",
    "v5d_two_tower": "two_tower",
    "v5e_multi_head": "multi_head",
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


def run_experiment(config_name, fusion_type, device, data_loader, feature_config, config):
    print(f"\n{'='*60}")
    print(f"融合策略: {config_name} ({fusion_type})")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    hierarchical_pairs = {
        "campaignid": "campaignsetid",
        "demand_pkgname": "business_type",
    }
    
    valid_pairs = {k: v for k, v in hierarchical_pairs.items()
                   if k in feature_config and v in feature_config}
    
    model = HierarchicalWideDeepV5(
        feature_config=feature_config,
        hierarchical_pairs=valid_pairs,
        fusion_type=fusion_type,
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
    
    return {
        "config_name": config_name,
        "fusion_type": fusion_type,
        "auc": results["auc"],
        "pcoc": results["pcoc"],
        "logloss": results["logloss"],
        "training_time_sec": training_time,
        "bt_grouped": results.get("bt_grouped", {}),
    }


def main():
    print("=" * 60)
    print("exp003 v5: Hierarchical 融合策略消融实验")
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
    
    baseline_results, feature_config, _ = load_shared_baseline()
    feature_config = {k: int(v) for k, v in feature_config.items()}
    
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    results_all = []
    for config_name, fusion_type in FUSION_CONFIGS.items():
        result = run_experiment(config_name, fusion_type, device, data_loader, feature_config, config)
        results_all.append(result)
    
    # 保存
    results_dir = Path("results/exp003_hierarchical_v5")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", "w") as f:
        json.dump({"experiment": "exp003_hierarchical_v5", "baseline": baseline_results, "results": results_all}, f, indent=2)
    
    # 打印
    print("\n" + "=" * 70)
    print("融合策略消融结果")
    print("=" * 70)
    print(f"{'配置':<20} {'融合类型':<15} {'AUC':<12} {'PCOC':<12} {'vs Baseline':<15}")
    print("-" * 70)
    
    if baseline_results:
        print(f"{'Baseline':<20} {'-':<15} {baseline_results['auc']:<12.4f} {baseline_results['pcoc']:<12.4f} {'-':<15}")
    
    for r in results_all:
        delta = r['auc'] - baseline_results['auc'] if baseline_results else 0
        marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
        print(f"{r['config_name']:<20} {r['fusion_type']:<15} {r['auc']:<12.4f} {r['pcoc']:<12.4f} {delta:+.4f} {marker}")
    
    print(f"\n✅ 完成！结果: {results_dir}/")


if __name__ == "__main__":
    main()
