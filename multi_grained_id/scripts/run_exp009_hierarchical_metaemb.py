#!/usr/bin/env python3
"""exp009: Hierarchical + MetaEmb 组合（集成已有模型）"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from src.data.loader import IVRDataLoader, collate_fn
from src.models.hierarchical import HierarchicalWideDeep
from src.models.metaemb import MetaEmbWideDeep
from src.utils.hardware_monitor import HardwareMonitor

HIERARCHICAL_PAIRS = {"campaignid": "campaignsetid", "offerid": "demand_pkgname", "demand_pkgname": "business_type"}

def evaluate_ensemble(models, dataloader, device):
    for m in models:
        m.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            features = {k: v.to(device) for k, v in batch["features"].items()}
            labels = batch["label"].cpu().numpy()
            preds_sum = sum(torch.sigmoid(m(features)).cpu().numpy() for m in models)
            all_preds.extend((preds_sum / len(models)).flatten())
            all_labels.extend(labels.flatten())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    return {"auc": float(roc_auc_score(all_labels, all_preds)), "pcoc": float(all_preds.mean() / all_labels.mean())}

def main():
    print("=" * 60)
    print("exp009: Hierarchical + MetaEmb 集成")
    print("=" * 60)
    
    config = {"label_col": "ctcvr_label", "batch_size": 512, "embedding_size": 64, "dnn_hidden_units": [256, 128]}
    
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
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # 加载 Hierarchical 模型
    hier_model = HierarchicalWideDeep(
        feature_config=feature_vocab_sizes,
        hierarchical_pairs=HIERARCHICAL_PAIRS,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
    ).to(device)
    
    hier_weights = Path("results/exp003_hierarchical/model.pt")
    if hier_weights.exists():
        hier_model.load_state_dict(torch.load(hier_weights, map_location=device))
        print(f"✅ 加载 Hierarchical 权重")
    
    # 加载 MetaEmb 模型
    meta_model = MetaEmbWideDeep(
        feature_config=feature_vocab_sizes,
        embedding_size=config["embedding_size"],
        dnn_hidden_units=config["dnn_hidden_units"],
        meta_features={"campaignid": ["campaignsetid", "business_type"], "demand_pkgname": ["business_type"]},
    ).to(device)
    
    meta_weights = Path("results/exp004_metaemb/model.pt")
    if meta_weights.exists():
        meta_model.load_state_dict(torch.load(meta_weights, map_location=device))
        print(f"✅ 加载 MetaEmb 权重")
    
    # 集成评估
    results = evaluate_ensemble([hier_model, meta_model], test_loader, device)
    
    output_dir = Path("results/exp009_hierarchical_metaemb")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ AUC={results['auc']:.6f}, PCOC={results['pcoc']:.4f}")

if __name__ == "__main__":
    main()
