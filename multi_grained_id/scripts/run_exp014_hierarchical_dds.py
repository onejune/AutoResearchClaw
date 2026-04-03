#!/usr/bin/env python3
"""exp014: Hierarchical + DDS 分层维度"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.data.loader import IVRDataLoader, collate_fn
from src.utils.hardware_monitor import HardwareMonitor

HIERARCHICAL_PAIRS = {"campaignid": "campaignsetid", "offerid": "demand_pkgname", "demand_pkgname": "business_type"}
EMBEDDING_DIMS = {"business_type": 32, "demand_pkgname": 64, "offerid": 64, "campaignsetid": 128, "campaignid": 128, "default": 64}

class DDSHierarchicalWideDeep(nn.Module):
    def __init__(self, feature_vocab_sizes, embedding_dims, hidden_dims=[256, 128], hierarchical_pairs=None):
        super().__init__()
        self.hierarchical_pairs = hierarchical_pairs or HIERARCHICAL_PAIRS
        self.embedding_dims = embedding_dims
        
        self.embeddings = nn.ModuleDict()
        self.emb_dims = {}
        for name, vocab_size in feature_vocab_sizes.items():
            dim = embedding_dims.get(name, embedding_dims.get("default", 64))
            self.emb_dims[name] = dim
            self.embeddings[name] = nn.Embedding(vocab_size + 1, dim, padding_idx=0)
        
        self.projections = nn.ModuleDict()
        for child, parent in self.hierarchical_pairs.items():
            child_dim = self.emb_dims.get(child, 64)
            parent_dim = self.emb_dims.get(parent, 32)
            target_dim = max(child_dim, parent_dim)
            self.projections[f"{child}_proj"] = nn.Linear(child_dim, target_dim)
            self.projections[f"{parent}_proj"] = nn.Linear(parent_dim, target_dim)
        
        self.final_dim = 128
        self.final_projections = nn.ModuleDict()
        for name in self.emb_dims:
            self.final_projections[name] = nn.Linear(self.emb_dims[name], self.final_dim)
        
        input_dim = len(feature_vocab_sizes) * self.final_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
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
                child_proj = self.projections[f"{child}_proj"](embeddings[child])
                parent_proj = self.projections[f"{parent}_proj"](embeddings[parent])
                embeddings[child] = child_proj + parent_proj
        
        final_embeddings = {}
        for name, emb in embeddings.items():
            final_embeddings[name] = self.final_projections[name](emb)
        
        emb = torch.cat([final_embeddings[name] for name in self.embeddings if name in final_embeddings], dim=-1)
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
            criterion(model(features), labels).backward()
            optimizer.step()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            features = {k: v.to(device) for k, v in batch["features"].items()}
            preds = torch.sigmoid(model(features)).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch["label"].cpu().numpy().flatten())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    return {"auc": float(roc_auc_score(all_labels, all_preds)), "pcoc": float(all_preds.mean() / all_labels.mean())}

def main():
    print("=" * 60)
    print("exp014: Hierarchical + DDS 分层维度")
    print("=" * 60)
    config = {"label_col": "ctcvr_label", "batch_size": 512, "lr": 5e-5, "epochs": 1, "embedding_dims": EMBEDDING_DIMS, "dnn_hidden_units": [256, 128]}
    monitor = HardwareMonitor()
    device, _ = monitor.select_device(min_memory_mb=4096)
    print(f"设备: {device}")
    print(f"分层维度: {EMBEDDING_DIMS}")
    with open(Path("results/shared_baseline/feature_config.json")) as f:
        feature_vocab_sizes = json.load(f)
    data_loader = IVRDataLoader(train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/", test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/", label_col=config["label_col"])
    train_dataset, test_dataset = data_loader.load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    model = DDSHierarchicalWideDeep(feature_vocab_sizes=feature_vocab_sizes, embedding_dims=config["embedding_dims"], hidden_dims=config["dnn_hidden_units"], hierarchical_pairs=HIERARCHICAL_PAIRS).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    results = train_and_evaluate(model, train_loader, test_loader, device, config)
    output_dir = Path("results/exp014_hierarchical_dds")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ AUC={results['auc']:.6f}, PCOC={results['pcoc']:.4f}")

if __name__ == "__main__":
    main()
