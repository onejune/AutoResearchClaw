#!/usr/bin/env python3
"""
exp003 v6: Hierarchical Embeddings - 多层级联消融

探索层级联的深度和结构：
- v6a: 单层级联（A→B）
- v6b: 双层级联（A→B→C）
- v6c: 三层级联（A→B→C→D）
- v6d: 并行层级（A→B, C→D 独立）
- v6e: 混合层级（级联 + 并行）
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
from src.utils.hardware_monitor import HardwareMonitor


class CascadeHierarchicalEmbedding(nn.Module):
    """级联层次化 Embedding：支持任意深度的层级链"""
    
    def __init__(self, vocab_sizes, embedding_dim=64):
        """
        Args:
            vocab_sizes: list of vocab sizes, from fine to coarse
        """
        super().__init__()
        self.num_levels = len(vocab_sizes)
        
        # 每层的 embedding
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab + 1, embedding_dim, padding_idx=0)
            for vocab in vocab_sizes
        ])
        
        # 层间门控（n-1 个门控）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            for _ in range(len(vocab_sizes) - 1)
        ])
        
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
    
    def forward(self, ids_list):
        """
        Args:
            ids_list: list of id tensors, from fine to coarse
        """
        # 从最粗粒度开始，逐层融合到细粒度
        current_emb = self.embeddings[-1](ids_list[-1])
        
        for i in range(self.num_levels - 2, -1, -1):
            fine_emb = self.embeddings[i](ids_list[i])
            gate = self.gates[i](torch.cat([fine_emb, current_emb], dim=-1))
            current_emb = gate * fine_emb + (1 - gate) * current_emb
        
        return current_emb


class HierarchicalWideDeepV6(nn.Module):
    """支持多层级联的 Hierarchical WideDeep"""
    
    def __init__(self, feature_config, cascade_chains, embedding_size=64,
                 dnn_hidden_units=[1024, 512, 256, 128], dropout=0.3):
        """
        Args:
            feature_config: {feature: vocab_size}
            cascade_chains: list of chains, each chain is [fine, ..., coarse]
        """
        super().__init__()
        
        self.feature_config = feature_config
        self.cascade_chains = cascade_chains
        
        # 识别级联特征和普通特征
        cascade_features = set()
        for chain in cascade_chains:
            cascade_features.update(chain)
        self.plain_features = [f for f in feature_config if f not in cascade_features]
        
        # 级联 Embedding
        self.cascade_embeddings = nn.ModuleList()
        self.cascade_feature_lists = []
        
        for chain in cascade_chains:
            valid_chain = [f for f in chain if f in feature_config]
            if len(valid_chain) >= 2:
                vocab_sizes = [feature_config[f] for f in valid_chain]
                self.cascade_embeddings.append(
                    CascadeHierarchicalEmbedding(vocab_sizes, embedding_size)
                )
                self.cascade_feature_lists.append(valid_chain)
        
        # 普通 Embedding
        self.plain_embeddings = nn.ModuleDict()
        for feat in self.plain_features:
            self.plain_embeddings[feat] = nn.Embedding(
                feature_config[feat] + 1, embedding_size, padding_idx=0
            )
        
        # DNN
        total_dim = (len(self.cascade_embeddings) + len(self.plain_features)) * embedding_size
        
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
        
        # 级联 embedding
        for cascade_emb, feat_list in zip(self.cascade_embeddings, self.cascade_feature_lists):
            ids_list = [features[f] for f in feat_list if f in features]
            if len(ids_list) == len(feat_list):
                embeddings.append(cascade_emb(ids_list))
        
        # 普通 embedding
        for feat, emb_layer in self.plain_embeddings.items():
            if feat in features:
                embeddings.append(emb_layer(features[feat]))
        
        x = torch.cat(embeddings, dim=-1)
        x = self.dnn(x)
        return self.output_layer(x).squeeze(-1)


# 不同的级联配置
CASCADE_CONFIGS = {
    "v6a_single": [
        ["demand_pkgname", "business_type"],  # 单层级联
    ],
    "v6b_double": [
        ["campaignid", "campaignsetid", "business_type"],  # 双层级联
    ],
    "v6c_triple": [
        ["campaignid", "campaignsetid", "offerid", "business_type"],  # 三层级联
    ],
    "v6d_parallel": [
        ["campaignid", "campaignsetid"],  # 并行链 1
        ["demand_pkgname", "business_type"],  # 并行链 2
    ],
    "v6e_mixed": [
        ["campaignid", "campaignsetid", "offerid"],  # 长链
        ["demand_pkgname", "business_type"],  # 短链
    ],
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


def run_experiment(config_name, cascade_chains, device, data_loader, feature_config, config):
    print(f"\n{'='*60}")
    print(f"级联配置: {config_name}")
    print(f"链结构: {cascade_chains}")
    print(f"{'='*60}")
    
    train_dataset, test_dataset = data_loader.load_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    model = HierarchicalWideDeepV6(
        feature_config=feature_config,
        cascade_chains=cascade_chains,
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
        "cascade_chains": cascade_chains,
        "auc": results["auc"],
        "pcoc": results["pcoc"],
        "logloss": results["logloss"],
        "training_time_sec": training_time,
        "bt_grouped": results.get("bt_grouped", {}),
    }


def main():
    print("=" * 60)
    print("exp003 v6: Hierarchical 多层级联消融实验")
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
    for config_name, chains in CASCADE_CONFIGS.items():
        result = run_experiment(config_name, chains, device, data_loader, feature_config, config)
        results_all.append(result)
    
    # 保存
    results_dir = Path("results/exp003_hierarchical_v6")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", "w") as f:
        json.dump({"experiment": "exp003_hierarchical_v6", "baseline": baseline_results, "results": results_all}, f, indent=2)
    
    # 打印
    print("\n" + "=" * 80)
    print("多层级联消融结果")
    print("=" * 80)
    print(f"{'配置':<15} {'链数':<8} {'最长链':<8} {'AUC':<12} {'PCOC':<12} {'vs Baseline':<15}")
    print("-" * 80)
    
    if baseline_results:
        print(f"{'Baseline':<15} {'-':<8} {'-':<8} {baseline_results['auc']:<12.4f} {baseline_results['pcoc']:<12.4f} {'-':<15}")
    
    for r in results_all:
        num_chains = len(r['cascade_chains'])
        max_len = max(len(c) for c in r['cascade_chains'])
        delta = r['auc'] - baseline_results['auc'] if baseline_results else 0
        marker = "🔺" if delta > 0.001 else ("🔻" if delta < -0.001 else "")
        print(f"{r['config_name']:<15} {num_chains:<8} {max_len:<8} {r['auc']:<12.4f} {r['pcoc']:<12.4f} {delta:+.4f} {marker}")
    
    print(f"\n✅ 完成！结果: {results_dir}/")


if __name__ == "__main__":
    main()
