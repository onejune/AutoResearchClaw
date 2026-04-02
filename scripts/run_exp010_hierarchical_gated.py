#!/usr/bin/env python3
"""
exp010: Hierarchical + 门控传递 + 残差连接
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss

from src.data.loader import IVRDataLoader, collate_fn
from src.utils.hardware_monitor import HardwareMonitor


HIERARCHICAL_PAIRS = {
    "campaignid": "campaignsetid",
    "offerid": "demand_pkgname",
    "demand_pkgname": "business_type",
}


class GatedHierarchicalWideDeep(nn.Module):
    """门控层级 WideDeep 模型"""
    
    def __init__(self, feature_vocab_sizes, embedding_dim=64, hidden_dims=[256, 128], hierarchical_pairs=None):
        super().__init__()
        
        self.hierarchical_pairs = hierarchical_pairs or HIERARCHICAL_PAIRS
        self.hierarchical_features = list(self.hierarchical_pairs.keys())
        
        # 基础 Embedding
        self.embeddings = nn.ModuleDict()
        for name, vocab_size in feature_vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0,
            )
        
        # 门控层
        self.gates = nn.ModuleDict()
        for child, parent in self.hierarchical_pairs.items():
            self.gates[f"{child}_gate"] = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid(),
            )
        
        # 残差权重
        self.residual_weights = nn.ParameterDict()
        for child in self.hierarchical_features:
            self.residual_weights[child] = nn.Parameter(torch.tensor(0.5))
        
        # DNN
        input_dim = len(feature_vocab_sizes) * embedding_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)
    
    def forward(self, features):
        embeddings = {}
        for name, indices in features.items():
            if name in self.embeddings:
                embeddings[name] = self.embeddings[name](indices)
        
        for child, parent in self.hierarchical_pairs.items():
            if child in embeddings and parent in embeddings:
                child_emb = embeddings[child]
                parent_emb = embeddings[parent]
                concat = torch.cat([child_emb, parent_emb], dim=-1)
                gate = self.gates[f"{child}_gate"](concat)
                fused = gate * child_emb + (1 - gate) * parent_emb
                res_w = torch.sigmoid(self.residual_weights[child])
                embeddings[child] = res_w * fused + (1 - res_w) * child_emb
        
        emb_list = [embeddings[name] for name in self.embeddings if name in embeddings]
        emb = torch.cat(emb_list, dim=-1)
        return self.dnn(emb).squeeze(-1)


def train_and_evaluate(model, train_loader, test_loader, device, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(config["epochs"]):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features = {k: v.to(device) for k, v in batch["features"].items()}
            labels = batch["label"].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            features = {k: v.to(device) for k, v in batch["features"].items()}
            labels = batch["label"].cpu().numpy()
            outputs = model(features)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        "auc": float(roc_auc_score(all_labels, all_preds)),
        "pcoc": float(all_preds.mean() / all_labels.mean()) if all_labels.mean() > 0 else float("nan"),
    }


def main():
    print("=" * 60)
    print("exp010: Hierarchical + 门控传递 + 残差连接")
    print("=" * 60)
    
    config = {"label_col": "ctcvr_label", "batch_size": 512, "lr": 5e-5, "epochs": 1, "embedding_size": 64, "dnn_hidden_units": [256, 128]}
    
    monitor = HardwareMonitor()
    device, _ = monitor.select_device(min_memory_mb=4096)
    print(f"设备: {device}")
    
    with open(Path("results/shared_baseline/feature_config.json")) as f:
        feature_vocab_sizes = json.load(f)
    
    data_loader = IVRDataLoader(
        train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/",
        test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/",
        label_col=config["label_col"],
    )
    
    train_dataset, test_dataset = data_loader.load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    model = GatedHierarchicalWideDeep(
        feature_vocab_sizes=feature_vocab_sizes,
        embedding_dim=config["embedding_size"],
        hidden_dims=config["dnn_hidden_units"],
        hierarchical_pairs=HIERARCHICAL_PAIRS,
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    results = train_and_evaluate(model, train_loader, test_loader, device, config)
    
    output_dir = Path("results/exp010_hierarchical_gated")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ AUC={results['auc']:.6f}, PCOC={results['pcoc']:.4f}")


if __name__ == "__main__":
    main()
