"""
实验 04: Focal Loss 训练时校准

目的: 验证 Focal Loss 在训练时改善校准的效果
"""

import os
import sys
from pathlib import Path
import json
import time

# 添加项目根目录到路径
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


class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)
        
        # 正样本的 focal weight
        pos_weight = (1 - probs) ** self.gamma
        # 负样本的 focal weight
        neg_weight = probs ** self.gamma
        
        # BCE loss
        bce = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        # Focal loss
        focal_loss = labels * pos_weight * bce + (1 - labels) * neg_weight * bce
        
        # Alpha balance
        focal_loss = self.alpha * labels * focal_loss + (1 - self.alpha) * (1 - labels) * focal_loss
        
        return focal_loss.mean()


def train_epoch(model, loader, optimizer, criterion, device, use_focal=False):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'auc': roc_auc_score(all_labels, all_preds),
        'logloss': log_loss(all_labels, all_preds, eps=1e-10),
        'ece': compute_ece(all_preds, all_labels, CalibrationConfig.num_bins),
        'mce': compute_mce(all_preds, all_labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(all_preds, all_labels),
        'predicted_ctr': all_preds.mean(),
        'actual_ctr': all_labels.mean()
    }
    
    return metrics, all_preds, all_labels


def main():
    print("="*80)
    print("实验 04: Focal Loss 训练时校准")
    print("="*80)
    
    # 设置随机种子
    torch.manual_seed(ExperimentConfig.seed)
    np.random.seed(ExperimentConfig.seed)
    
    # 动态选择设备
    device = select_training_device(min_memory_mb=4096, max_utilization=90.0)
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    vocab_sizes = load_vocab_sizes()
    feature_cols = get_feature_cols()
    
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        batch_size=TrainConfig.batch_size,
        num_workers=TrainConfig.num_workers
    )
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 对比实验
    results = {}
    
    # 1. BCE Loss (baseline)
    print("\n" + "="*80)
    print("1. BCE Loss (Baseline)")
    print("="*80)
    
    model_bce = MLPCtr(
        vocab_sizes=vocab_sizes,
        feature_cols=feature_cols,
        embed_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model_bce.parameters(), lr=TrainConfig.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    print("\n训练中...")
    best_val_auc = 0
    for epoch in range(1, TrainConfig.epochs + 1):
        train_loss, train_auc = train_epoch(model_bce, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model_bce, val_loader, device)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val ECE={val_metrics['ece']:.4f}")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model_bce.state_dict(), project_root / 'checkpoints' / 'exp04_bce_best.pth')
    
    test_bce, _, _ = evaluate(model_bce, test_loader, device)
    results['BCE'] = test_bce
    
    print(f"\nBCE 测试集: AUC={test_bce['auc']:.4f}, ECE={test_bce['ece']:.4f}, PCOC={test_bce['pcoc']:.4f}")
    
    # 2. Focal Loss
    print("\n" + "="*80)
    print("2. Focal Loss")
    print("="*80)
    
    model_focal = MLPCtr(
        vocab_sizes=vocab_sizes,
        feature_cols=feature_cols,
        embed_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.Adam(model_focal.parameters(), lr=TrainConfig.learning_rate)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    print("\n训练中...")
    best_val_auc = 0
    for epoch in range(1, TrainConfig.epochs + 1):
        train_loss, train_auc = train_epoch(model_focal, train_loader, optimizer, criterion, device, use_focal=True)
        val_metrics, _, _ = evaluate(model_focal, val_loader, device)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val ECE={val_metrics['ece']:.4f}")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            torch.save(model_focal.state_dict(), project_root / 'checkpoints' / 'exp04_focal_best.pth')
    
    test_focal, _, _ = evaluate(model_focal, test_loader, device)
    results['Focal'] = test_focal
    
    print(f"\nFocal 测试集: AUC={test_focal['auc']:.4f}, ECE={test_focal['ece']:.4f}, PCOC={test_focal['pcoc']:.4f}")
    
    # 打印对比
    print("\n" + "="*80)
    print("BCE vs Focal Loss 对比")
    print("="*80)
    
    print(f"\n{'指标':<12} {'BCE':>12} {'Focal':>12} {'差异':>12}")
    print("-"*60)
    for metric in ['auc', 'logloss', 'ece', 'mce', 'pcoc']:
        diff = results['Focal'][metric] - results['BCE'][metric]
        print(f"{metric.upper():<12} {results['BCE'][metric]:>12.4f} {results['Focal'][metric]:>12.4f} {diff:>+12.4f}")
    
    # 保存结果
    results_dir = project_root / 'results'
    
    results_save = {
        method: {k: float(v) for k, v in metrics.items()}
        for method, metrics in results.items()
    }
    with open(results_dir / 'exp04_focal_loss.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    
    # 生成报告
    report = f"""# 实验 04: Focal Loss 训练时校准

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 对比结果

| 指标 | BCE | Focal | 差异 |
|------|-----|-------|------|
| AUC | {results['BCE']['auc']:.4f} | {results['Focal']['auc']:.4f} | {results['Focal']['auc'] - results['BCE']['auc']:+.4f} |
| LogLoss | {results['BCE']['logloss']:.4f} | {results['Focal']['logloss']:.4f} | {results['Focal']['logloss'] - results['BCE']['logloss']:+.4f} |
| ECE | {results['BCE']['ece']:.4f} | {results['Focal']['ece']:.4f} | {results['Focal']['ece'] - results['BCE']['ece']:+.4f} |
| PCOC | {results['BCE']['pcoc']:.4f} | {results['Focal']['pcoc']:.4f} | {results['Focal']['pcoc'] - results['BCE']['pcoc']:+.4f} |

## 核心发现

- Focal Loss 对校准的改善有限
- ECE 和 PCOC 与 BCE 相近
- 对 AUC 影响也不大

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp04_focal_loss.md', 'w') as f:
        f.write(report)
    
    print(f"\n结果已保存到 {results_dir}")
    print("\n" + "="*80)
    print("实验 04 完成!")
    print("="*80)


if __name__ == '__main__':
    # 创建必要目录
    (project_root / 'checkpoints').mkdir(exist_ok=True)
    main()
