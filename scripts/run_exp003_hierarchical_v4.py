#!/usr/bin/env python3
"""
exp003 v4: Hierarchical Embeddings - 门控机制消融

探索不同的门控网络设计：
- v4a: 简单门控（单层 sigmoid）
- v4b: 深度门控（3 层 MLP）
- v4c: 注意力门控（self-attention）
- v4d: 频次感知门控（根据 ID 频次调整）
"""

import os
import sys
import json
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.loader import IVRDataLoader, collate_fn
from src.train.trainer import DynamicTrainer
from src.utils.hardware_monitor import HardwareMonitor


class SimpleGateHierarchicalEmbedding(nn.Module):
    """简单门控：单层 sigmoid"""
    
    def __init__(self, fine_vocab_size, coarse_vocab_size, embedding_dim=64):
        super().__init__()
        self.fine_embedding = nn.Embedding(fine_vocab_size + 1, embedding_dim, padding_idx=0)
        self.coarse_embedding = nn.Embedding(coarse_vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 简单门控：直接用 embedding 的内积
        self.gate_weight = nn.Parameter(torch.zeros(embedding_dim))
        
        nn.init.xavier_uniform_(self.fine_embedding.weight)
        nn.init.xavier_uniform_(self.coarse_embedding.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine_emb = self.fine_embedding(fine_ids)
        coarse_emb = self.coarse_embedding(coarse_ids)
        
        # 简单门控：基于 fine embedding 和可学习权重
        gate = torch.sigmoid((fine_emb * self.gate_weight).sum(dim=-1, keepdim=True))
        
        fused = gate * fine_emb + (1 - gate) * coarse_emb
        return fused, gate


class DeepGateHierarchicalEmbedding(nn.Module):
    """深度门控：3 层 MLP"""
    
    def __init__(self, fine_vocab_size, coarse_vocab_size, embedding_dim=64):
        super().__init__()
        self.fine_embedding = nn.Embedding(fine_vocab_size + 1, embedding_dim, padding_idx=0)
        self.coarse_embedding = nn.Embedding(coarse_vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 深度门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.fine_embedding.weight)
        nn.init.xavier_uniform_(self.coarse_embedding.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine_emb = self.fine_embedding(fine_ids)
        coarse_emb = self.coarse_embedding(coarse_ids)
        
        combined = torch.cat([fine_emb, coarse_emb], dim=-1)
        gate = self.gate_network(combined)
        
        fused = gate * fine_emb + (1 - gate) * coarse_emb
        return fused, gate


class AttentionGateHierarchicalEmbedding(nn.Module):
    """注意力门控：cross-attention"""
    
    def __init__(self, fine_vocab_size, coarse_vocab_size, embedding_dim=64, num_heads=4):
        super().__init__()
        self.fine_embedding = nn.Embedding(fine_vocab_size + 1, embedding_dim, padding_idx=0)
        self.coarse_embedding = nn.Embedding(coarse_vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 注意力层
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.scale = math.sqrt(embedding_dim)
        
        # 输出门控
        self.gate_proj = nn.Linear(embedding_dim, 1)
        
        nn.init.xavier_uniform_(self.fine_embedding.weight)
        nn.init.xavier_uniform_(self.coarse_embedding.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine_emb = self.fine_embedding(fine_ids)
        coarse_emb = self.coarse_embedding(coarse_ids)
        
        # Cross-attention: fine 作为 query，coarse 作为 key/value
        q = self.query(fine_emb)
        k = self.key(coarse_emb)
        v = self.value(coarse_emb)
        
        # 计算注意力权重
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / self.scale)
        
        # 加权融合
        attended = attn * fine_emb + (1 - attn) * v
        
        # 最终门控
        gate = torch.sigmoid(self.gate_proj(attended))
        fused = gate * fine_emb + (1 - gate) * coarse_emb
        
        return fused, gate


class FrequencyAwareHierarchicalEmbedding(nn.Module):
    """频次感知门控：低频 ID 偏向粗粒度"""
    
    def __init__(self, fine_vocab_size, coarse_vocab_size, embedding_dim=64):
        super().__init__()
        self.fine_embedding = nn.Embedding(fine_vocab_size + 1, embedding_dim, padding_idx=0)
        self.coarse_embedding = nn.Embedding(coarse_vocab_size + 1, embedding_dim, padding_idx=0)
        
        # 频次 embedding（可学习的频次权重）
        self.freq_embedding = nn.Embedding(fine_vocab_size + 1, 1, padding_idx=0)
        nn.init.ones_(self.freq_embedding.weight)  # 初始化为 1
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.fine_embedding.weight)
        nn.init.xavier_uniform_(self.coarse_embedding.weight)
    
    def forward(self, fine_ids, coarse_ids):
        fine_emb = self.fine_embedding(fine_ids)
        coarse_emb = self.coarse_embedding(coarse_ids)
        
        # 获取频次权重
        freq_weight = torch.sigmoid(self.freq_embedding(fine_ids))  # (batch, 1)
        
        # 拼接 embedding 和频次
        combined = torch.cat([fine_emb, coarse_emb, freq_weight], dim=-1)
        gate = self.gate_network(combined)
        
        # 频次调整：低频 ID 的 gate 偏向 0（使用粗粒度）
        adjusted_gate = gate * freq_weight
        
        fused = adjusted_gate * fine_emb + (1 - adjusted_gate) * coarse_emb
        return fused, adjusted_gate


class HierarchicalWideDeepV4(nn.Module):
    """支持多种门控机制的 Hierarchical WideDeep"""
    
    def __init__(
        self,
        feature_config,
        hierarchical_pairs,
        gate_type="standard",  # simple, deep, attention, frequency
        embedding_size=64,
        dnn_hidden_units=[1024, 512, 256, 128],
        dropout=0.3
    ):
        super().__init__()
        
        self.feature_config = feature_config
        self.hierarchical_pairs = hierarchical_pairs
        self.gate_type = gate_type
        
        # 选择门控类型
        gate_classes = {
            "simple": SimpleGateHierarchicalEmbedding,
            "deep": DeepGateHierarchicalEmbedding,
            "attention": AttentionGateHierarchicalEmbedding,
            "frequency": FrequencyAwareHierarchicalEmbedding,
        }
        GateClass = gate_classes.get(gate_type, DeepGateHierarchicalEmbedding)
        
        # 层次化特征
        self.hier_fine = set(hierarchical_pairs.keys())
        self.hier_coarse = set(hierarchical_pairs.values())
        self.plain_features = [f for f in feature_config if f not in self.hier_fine and f not in self.hier_coarse]
        
        # 层次化 Embedding
        self.hierarchical_embeddings = nn.ModuleDict()
        for fine, coarse in hierarchical_pairs.items():
            if fine in feature_config and coarse in feature_config:
                self.hierarchical_embeddings[fine] = GateClass(
                    feature_config[fine], feature_config[coarse], embedding_size
                )
        
        # 普通 Embedding
        self.plain_embeddings = nn.ModuleDict()
        for feat in self.plain_features:
            self.plain_embeddings[feat] = nn.Embedding(
                feature_config[feat] + 1, embedding_size, padding_idx=0
            )
        
        # DNN
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
        gate_stats = {}
        
        for fine, hier_emb in self.hierarchical_embeddings.items():
            coarse = self.hierarchical_pairs[fine]
            if fine in features and coarse in features:
                emb, gate = hier_emb(features[fine], features[coarse])
                embeddings.append(emb)
                gate_stats[fine] = gate.mean().item()
        
        for feat, emb_layer in self.plain_embeddings.items():
            if feat in features:
                embeddings.append(emb_layer(features[feat]))
        
        x = torch.cat(embeddings, dim=-1)
        x = self.dnn(x)
        logits = self.output_layer(x).squeeze(-1)
        
        return logits
    
    def get_gate_statistics(self):
        return {"gate_type": self.gate_type}


GATE_CONFIGS = {
    "v4a_simple": "simple",
    "v4b_deep": "deep",
    "v4c_attention": "attention",
    "v4d_frequency": "frequency",
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


def run_experiment(config_name, gate_type, device, data_loader, feature_config, config):
    print(f"\n{'='*60}")
    print(f"门控类型: {config_name} ({gate_type})")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    # 使用最优层次化配置
    hierarchical_pairs = {
        "campaignid": "campaignsetid",
        "demand_pkgname": "business_type",
    }
    
    valid_pairs = {k: v for k, v in hierarchical_pairs.items()
                   if k in feature_config and v in feature_config}
    
    model = HierarchicalWideDeepV4(
        feature_config=feature_config,
        hierarchical_pairs=valid_pairs,
        gate_type=gate_type,
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
        "gate_type": gate_type,
        "auc": results["auc"],
        "pcoc": results["pcoc"],
        "logloss": results["logloss"],
        "training_time_sec": training_time,
        "bt_grouped": results.get("bt_grouped", {}),
    }


def main():
    print("=" * 60)
    print("exp003 v4: Hierarchical 门控机制消融实验")
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
    
    baseline_results, feature_config, baseline_bt = load_shared_baseline()
    feature_config = {k: int(v) for k, v in feature_config.items()}
    
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    results_all = []
    for config_name, gate_type in GATE_CONFIGS.items():
        result = run_experiment(config_name, gate_type, device, data_loader, feature_config, config)
        results_all.append(result)
    
    # 保存
    results_dir = Path("results/exp003_hierarchical_v4")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        "experiment": "exp003_hierarchical_v4",
        "baseline": baseline_results,
        "results": results_all,
    }
    
    with open(results_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("门控机制消融结果")
    print("=" * 70)
    print(f"{'配置':<20} {'门控类型':<15} {'AUC':<12} {'PCOC':<12} {'vs Baseline':<15}")
    print("-" * 70)
    
    if baseline_results:
        print(f"{'Baseline':<20} {'-':<15} {baseline_results['auc']:<12.4f} {baseline_results['pcoc']:<12.4f} {'-':<15}")
    
    for r in results_all:
        delta = r['auc'] - baseline_results['auc'] if baseline_results else 0
        marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
        print(f"{r['config_name']:<20} {r['gate_type']:<15} {r['auc']:<12.4f} {r['pcoc']:<12.4f} {delta:+.4f} {marker}")
    
    print(f"\n✅ 完成！结果: {results_dir}/")


if __name__ == "__main__":
    main()
