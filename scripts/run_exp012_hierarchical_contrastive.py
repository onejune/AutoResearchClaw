#!/usr/bin/env python3
"""exp012: Hierarchical + 对比学习"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.data.loader import IVRDataLoader, collate_fn
from src.utils.hardware_monitor import HardwareMonitor

HIERARCHICAL_PAIRS = {"campaignid": "campaignsetid", "offerid": "demand_pkgname", "demand_pkgname": "business_type"}

class ContrastiveHierarchicalWideDeep(nn.Module):
    def __init__(self, feature_vocab_sizes, embedding_dim=64, hidden_dims=[256, 128], hierarchical_pairs=None, temperature=0.1):
        super().__init__()
        self.hierarchical_pairs = hierarchical_pairs or HIERARCHICAL_PAIRS
        self.temperature = temperature
        
        self.embeddings = nn.ModuleDict()
        for name, vocab_size in feature_vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        
        self.projections = nn.ModuleDict()
        for child, parent in self.hierarchical_pairs.items():
            self.projections[f"{child}_proj"] = nn.Linear(embedding_dim, embedding_dim)
        
        input_dim = len(feature_vocab_sizes) * embedding_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.dnn = nn.Sequential(*layers)
    
    def forward(self, features, return_hier_emb=False):
        embeddings = {}
        hier_emb = {}
        for name, indices in features.items():
            if name in self.embeddings:
                emb = self.embeddings[name](indices)
                embeddings[name] = emb
                hier_emb[name] = emb.clone()
        for child, parent in self.hierarchical_pairs.items():
            if child in embeddings and parent in embeddings:
                embeddings[child] = self.projections[f"{child}_proj"](embeddings[child]) + embeddings[parent]
        emb = torch.cat([embeddings[name] for name in self.embeddings if name in embeddings], dim=-1)
        return (emb, hier_emb) if return_hier_emb else emb
    
    def contrastive_loss(self, hier_emb):
        loss, count = 0.0, 0
        for child, parent in self.hierarchical_pairs.items():
            if child in hier_emb and parent in hier_emb:
                child_norm = F.normalize(hier_emb[child], dim=-1)
                parent_norm = F.normalize(hier_emb[parent], dim=-1)
                neg_sim = torch.matmul(child_norm, parent_norm.T) / self.temperature
                labels = torch.arange(child_norm.size(0), device=child_norm.device)
                loss += F.cross_entropy(neg_sim, labels)
                count += 1
        return loss / max(count, 1)

def train_and_evaluate(model, train_loader, test_loader, device, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCEWithLogitsLoss()
    cl_weight = config.get("contrastive_weight", 0.1)
    model.train()
    for epoch in range(config["epochs"]):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features = {k: v.to(device) for k, v in batch["features"].items()}
            labels = batch["label"].float().to(device)
            optimizer.zero_grad()
            emb, hier_emb = model.forward(features, return_hier_emb=True)
            outputs = model.dnn(emb).squeeze(-1)
            loss = criterion(outputs, labels) + cl_weight * model.contrastive_loss(hier_emb)
            loss.backward()
            optimizer.step()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            features = {k: v.to(device) for k, v in batch["features"].items()}
            preds = torch.sigmoid(model.dnn(model.forward(features))).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch["label"].cpu().numpy().flatten())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    return {"auc": float(roc_auc_score(all_labels, all_preds)), "pcoc": float(all_preds.mean() / all_labels.mean())}

def main():
    print("=" * 60)
    print("exp012: Hierarchical + Contrastive")
    print("=" * 60)
    config = {"label_col": "ctcvr_label", "batch_size": 512, "lr": 5e-5, "epochs": 1, "embedding_size": 64, "dnn_hidden_units": [256, 128], "contrastive_weight": 0.1}
    monitor = HardwareMonitor()
    device, _ = monitor.select_device(min_memory_mb=4096)
    print(f"设备: {device}")
    with open(Path("results/shared_baseline/feature_config.json")) as f:
        feature_vocab_sizes = json.load(f)
    data_loader = IVRDataLoader(train_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/", test_path="/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/", label_col=config["label_col"])
    train_dataset, test_dataset = data_loader.load_datasets()
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    model = ContrastiveHierarchicalWideDeep(feature_vocab_sizes=feature_vocab_sizes, embedding_dim=config["embedding_size"], hidden_dims=config["dnn_hidden_units"], hierarchical_pairs=HIERARCHICAL_PAIRS).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    results = train_and_evaluate(model, train_loader, test_loader, device, config)
    output_dir = Path("results/exp012_hierarchical_contrastive")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ AUC={results['auc']:.6f}, PCOC={results['pcoc']:.4f}")

if __name__ == "__main__":
    main()
