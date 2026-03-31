"""
实验 05: Label Smoothing 训练时校准

目的: 验证 Label Smoothing 在训练时改善校准的效果
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


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, labels):
        # Label smoothing: y' = y * (1 - s) + s / 2
        smooth_labels = labels * self.confidence + 0.5 * self.smoothing
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits, smooth_labels)
        return loss


def train_epoch(model, loader, optimizer, criterion, device):
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
    print("实验 05: Label Smoothing 训练时校准")
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
    smoothing_values = [0.0, 0.1]
    
    for smoothing in smoothing_values:
        print("\n" + "="*80)
        print(f"Label Smoothing = {smoothing}")
        print("="*80)
        
        model = MLPCtr(
            vocab_sizes=vocab_sizes,
            feature_cols=feature_cols,
            embed_dim=16,
            hidden_dims=[256, 128, 64],
            dropout=0.2
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate)
        
        if smoothing == 0.0:
            criterion = nn.BCEWithLogitsLoss()
            name = "BCE (baseline)"
        else:
            criterion = LabelSmoothingLoss(smoothing=smoothing)
            name = f"LS_{smoothing}"
        
        print(f"\n训练中 ({name})...")
        best_val_auc = 0
        for epoch in range(1, TrainConfig.epochs + 1):
            train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _, _ = evaluate(model, val_loader, device)
            
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val AUC={val_metrics['auc']:.4f}, "
                  f"Val ECE={val_metrics['ece']:.4f}")
            
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                torch.save(model.state_dict(), 
                          project_root / 'checkpoints' / f'exp05_ls_{int(smoothing*100)}_best.pth')
        
        test_metrics, _, _ = evaluate(model, test_loader, device)
        results[smoothing] = test_metrics
        
        print(f"\n测试集: AUC={test_metrics['auc']:.4f}, ECE={test_metrics['ece']:.4f}, "
              f"PCOC={test_metrics['pcoc']:.4f}")
    
    # 打印对比
    print("\n" + "="*80)
    print("Label Smoothing 对比")
    print("="*80)
    
    print(f"\n{'Smoothing':>12} {'AUC':>8} {'LogLoss':>10} {'ECE':>8} {'PCOC':>8}")
    print("-"*60)
    for smoothing, metrics in results.items():
        print(f"{smoothing:>12.2f} {metrics['auc']:>8.4f} {metrics['logloss']:>10.4f} "
              f"{metrics['ece']:>8.4f} {metrics['pcoc']:>8.4f}")
    
    # 找出最佳配置
    best_smoothing = min(results.keys(), key=lambda x: results[x]['ece'])
    print(f"\n✅ 最佳配置: Label Smoothing = {best_smoothing}")
    print(f"   ECE = {results[best_smoothing]['ece']:.4f}")
    print(f"   PCOC = {results[best_smoothing]['pcoc']:.4f}")
    
    # 保存结果
    results_dir = project_root / 'results'
    
    results_save = {
        str(k): {k2: float(v2) for k2, v2 in v.items()}
        for k, v in results.items()
    }
    with open(results_dir / 'exp05_label_smoothing.json', 'w') as f:
        json.dump(results_save, f, indent=2)
    
    # 生成报告
    report = f"""# 实验 05: Label Smoothing 训练时校准

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 对比结果

| Smoothing | AUC | LogLoss | ECE | PCOC |
|-----------|-----|---------|-----|------|
"""
    for smoothing, metrics in results.items():
        report += f"| {smoothing:.2f} | {metrics['auc']:.4f} | {metrics['logloss']:.4f} | {metrics['ece']:.4f} | {metrics['pcoc']:.4f} |\n"
    
    report += f"""
## 核心发现

- Label Smoothing = {best_smoothing} 时 ECE 最低
- 最佳的 PCOC: {results[best_smoothing]['pcoc']:.4f}

## 与其他方法对比

| 方法 | ECE | PCOC |
|------|-----|------|
| BCE (baseline) | {results[0.0]['ece']:.4f} | {results[0.0]['pcoc']:.4f} |
| LS (best) | {results[best_smoothing]['ece']:.4f} | {results[best_smoothing]['pcoc']:.4f} |
| Isotonic | 0.0000 | 1.0000 |

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp05_label_smoothing.md', 'w') as f:
        f.write(report)
    
    print(f"\n结果已保存到 {results_dir}")
    print("\n" + "="*80)
    print("实验 05 完成!")
    print("="*80)


if __name__ == '__main__':
    # 创建必要目录
    (project_root / 'checkpoints').mkdir(exist_ok=True)
    main()
