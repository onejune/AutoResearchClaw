"""
实验 06: 按 business_type 分组训练 - 简化版

目的：快速验证分组训练对校准的影响
"""

import os, sys
from pathlib import Path
import json, time
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from config import TrainConfig, CalibrationConfig, ExperimentConfig, get_data_path
from data.dataset import create_dataloaders, get_feature_cols, load_vocab_sizes, IVRDataset
from models.mlp import MLPCtr
from calibration.methods import compute_ece, compute_mce, compute_pcoc
from training.hardware_monitor import select_training_device


def main():
    print("="*80)
    print("实验 06: 按 business_type 分组训练")
    print("="*80)
    
    torch.manual_seed(ExperimentConfig.seed)
    device = select_training_device(min_memory_mb=4096, max_utilization=90.0)
    print(f"\n使用设备：{device}")
    
    # 加载数据
    print("\n加载数据...")
    vocab_sizes = load_vocab_sizes()
    feature_cols = get_feature_cols()
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        batch_size=TrainConfig.batch_size, num_workers=TrainConfig.num_workers, label_col='ctcvr_label'
    )
    
    # 重新加载完整数据集用于分组
    full_dataset = IVRDataset(
        get_data_path('train'), 
        vocab_sizes, 
        feature_cols,
        label_col='ctcvr_label'
    )
    
    df = pd.DataFrame({'business_type': full_dataset.business_types, 'label': full_dataset.labels})
    top_bts = df['business_type'].value_counts().head(3).index.tolist()
    print(f"Top 3 business_type: {top_bts} (占 {(df['business_type'].isin(top_bts)).sum()/len(df)*100:.1f}%)")
    
    results = {}
    
    # 整体模型
    print("\n1. 整体模型...")
    model = MLPCtr(vocab_sizes=vocab_sizes, feature_cols=feature_cols,
                  embed_dim=16, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate)
    
    for epoch in range(TrainConfig.epochs):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(features), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(features)).cpu().numpy()
            print(f"  Epoch {epoch+1}: AUC={roc_auc_score(labels.cpu().numpy(), preds):.4f}")
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            preds = torch.sigmoid(model(features.to(device))).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    results['overall'] = {
        'auc': roc_auc_score(all_labels, all_preds),
        'ece': compute_ece(all_preds, all_labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(all_preds, all_labels)
    }
    print(f"  测试集：AUC={results['overall']['auc']:.4f}, ECE={results['overall']['ece']:.4f}, PCOC={results['overall']['pcoc']:.4f}")
    
    # 分组训练 (只训练 BT1)
    print(f"\n2. 分组训练 (BT{top_bts[0]})...")
    bt = top_bts[0]
    mask = np.array(full_dataset.business_types) == bt
    indices = np.where(mask)[0]
    print(f"  样本数：{len(indices)}")
    
    if len(indices) > 10000:
        from torch.utils.data import Subset
        train_subset = Subset(full_dataset, indices[:int(len(indices)*0.8)])
        val_subset = Subset(full_dataset, indices[int(len(indices)*0.8):])
        
        train_bt = torch.utils.data.DataLoader(train_subset, batch_size=min(TrainConfig.batch_size, 512), shuffle=True)
        val_bt = torch.utils.data.DataLoader(val_subset, batch_size=min(TrainConfig.batch_size, 512))
        
        model_bt = MLPCtr(vocab_sizes=vocab_sizes, feature_cols=feature_cols,
                         embed_dim=16, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
        optimizer_bt = torch.optim.Adam(model_bt.parameters(), lr=TrainConfig.learning_rate)
        
        for epoch in range(TrainConfig.epochs):
            model_bt.train()
            for features, labels in train_bt:
                features, labels = features.to(device), labels.to(device)
                optimizer_bt.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model_bt(features), labels)
                loss.backward()
                optimizer_bt.step()
            
            model_bt.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model_bt(features)).cpu().numpy()
                print(f"  Epoch {epoch+1}: AUC={roc_auc_score(labels.cpu().numpy(), preds):.4f}")
        
        model_bt.eval()
        preds_bt, labels_bt = [], []
        with torch.no_grad():
            for features, labels in val_bt:
                preds = torch.sigmoid(model_bt(features.to(device))).cpu().numpy()
                preds_bt.extend(preds)
                labels_bt.extend(labels.numpy())
        
        preds_bt, labels_bt = np.array(preds_bt), np.array(labels_bt)
        results[f'bt_{bt}'] = {
            'auc': roc_auc_score(labels_bt, preds_bt),
            'ece': compute_ece(preds_bt, labels_bt, CalibrationConfig.num_bins),
            'pcoc': compute_pcoc(preds_bt, labels_bt)
        }
        print(f"  验证集：AUC={results[f'bt_{bt}']['auc']:.4f}, ECE={results[f'bt_{bt}']['ece']:.4f}, PCOC={results[f'bt_{bt}']['pcoc']:.4f}")
    
    # 对比
    print("\n" + "="*80)
    print("分组训练 vs 整体训练")
    print("="*80)
    print(f"{'Model':>12} {'AUC':>8} {'ECE':>8} {'PCOC':>8}")
    print("-"*40)
    for name, m in results.items():
        print(f"{name:>12} {m['auc']:>8.4f} {m['ece']:>8.4f} {m['pcoc']:>8.4f}")
    
    # 保存
    with open(project_root / 'results' / 'exp06_grouped_training.json', 'w') as f:
        json.dump({k: {k2: float(v2) for k2, v2 in v.items()} for k, v in results.items()}, f, indent=2)
    
    print(f"\n✅ 结果已保存")


if __name__ == '__main__':
    main()
