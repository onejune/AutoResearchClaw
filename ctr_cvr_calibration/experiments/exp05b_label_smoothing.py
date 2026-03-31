"""
实验 05B: 快速 Label Smoothing 对比

目的: 快速验证 Label Smoothing 效果
"""

import os
import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from config import TrainConfig, CalibrationConfig, ExperimentConfig
from data.dataset import create_dataloaders, get_feature_cols, load_vocab_sizes
from models.mlp import MLPCtr
from calibration.methods import compute_ece, compute_mce, compute_pcoc
from training.hardware_monitor import select_training_device


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, labels):
        smooth_labels = labels * self.confidence + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, smooth_labels)


def train_and_eval(model, train_loader, val_loader, test_loader, criterion, device, name):
    print(f"\n训练 {name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate)
    
    best_val_auc = 0
    for epoch in range(1, TrainConfig.epochs + 1):
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                logits = model(features)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), project_root / 'checkpoints' / f'{name}.pth')
    
    # 测试
    model.load_state_dict(torch.load(project_root / 'checkpoints' / f'{name}.pth'))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    metrics = {
        'auc': roc_auc_score(test_labels, test_preds),
        'logloss': log_loss(test_labels, test_preds, eps=1e-10),
        'ece': compute_ece(test_preds, test_labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(test_preds, test_labels)
    }
    
    print(f"  AUC={metrics['auc']:.4f}, ECE={metrics['ece']:.4f}, PCOC={metrics['pcoc']:.4f}")
    return metrics


def main():
    print("="*60)
    print("实验 05B: Label Smoothing 快速对比")
    print("="*60)
    
    torch.manual_seed(ExperimentConfig.seed)
    np.random.seed(ExperimentConfig.seed)
    
    device = select_training_device(min_memory_mb=4096, max_utilization=90.0)
    print(f"\n使用设备: {device}")
    
    print("\n加载数据...")
    vocab_sizes = load_vocab_sizes()
    feature_cols = get_feature_cols()
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        batch_size=TrainConfig.batch_size,
        num_workers=TrainConfig.num_workers
    )
    
    results = {}
    
    # Baseline (BCE)
    model = MLPCtr(vocab_sizes=vocab_sizes, feature_cols=feature_cols,
                  embed_dim=16, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    results['BCE'] = train_and_eval(model, train_loader, val_loader, test_loader,
                                    nn.BCEWithLogitsLoss(), device, 'exp05_bce')
    
    # Label Smoothing
    model = MLPCtr(vocab_sizes=vocab_sizes, feature_cols=feature_cols,
                  embed_dim=16, hidden_dims=[256, 128, 64], dropout=0.2).to(device)
    results['LS_0.1'] = train_and_eval(model, train_loader, val_loader, test_loader,
                                       LabelSmoothingLoss(smoothing=0.1), device, 'exp05_ls')
    
    # 对比
    print("\n" + "="*60)
    print("Label Smoothing 对比")
    print("="*60)
    print(f"{'Method':>12} {'AUC':>8} {'ECE':>8} {'PCOC':>8}")
    print("-"*40)
    for name, m in results.items():
        print(f"{name:>12} {m['auc']:>8.4f} {m['ece']:>8.4f} {m['pcoc']:>8.4f}")
    
    # 保存
    results_dir = project_root / 'results'
    with open(results_dir / 'exp05_label_smoothing.json', 'w') as f:
        json.dump({k: {k2: float(v2) for k2, v2 in v.items()} for k, v in results.items()}, f, indent=2)
    
    print(f"\n✅ 结果已保存")


if __name__ == '__main__':
    main()
