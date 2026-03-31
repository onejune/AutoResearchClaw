"""
实验 01: 基线模型校准性能评估

目的: 评估未校准模型的校准性能
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


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (features, labels) in enumerate(loader):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % TrainConfig.log_interval == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, auc


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits)
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    metrics = {
        'auc': roc_auc_score(all_labels, all_preds),
        'logloss': log_loss(all_labels, all_preds, eps=1e-10),
        'ece': compute_ece(all_preds, all_labels, CalibrationConfig.num_bins),
        'mce': compute_mce(all_preds, all_labels, CalibrationConfig.num_bins),
        'pcoc': compute_pcoc(all_preds, all_labels),
        'predicted_ctr': all_preds.mean(),
        'actual_ctr': all_labels.mean(),
    }
    
    return metrics, all_preds, all_labels, all_logits


def print_metrics(metrics: dict, prefix: str = ""):
    """打印指标"""
    print(f"\n{prefix}校准指标:")
    print(f"  AUC:    {metrics['auc']:.4f}")
    print(f"  LogLoss: {metrics['logloss']:.4f}")
    print(f"  ECE:    {metrics['ece']:.4f}")
    print(f"  MCE:    {metrics['mce']:.4f}")
    print(f"  PCOC:   {metrics['pcoc']:.4f}")
    print(f"  预测 CTR: {metrics['predicted_ctr']:.4f}")
    print(f"  实际 CTR: {metrics['actual_ctr']:.4f}")


def main():
    print("="*60)
    print("实验 01: 基线模型校准性能评估")
    print("="*60)
    
    # 设置随机种子
    torch.manual_seed(ExperimentConfig.seed)
    np.random.seed(ExperimentConfig.seed)
    
    # 🎯 动态选择设备（优先 GPU）
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
    
    # 创建模型
    print("\n创建模型...")
    model = MLPCtr(
        vocab_sizes=vocab_sizes,
        feature_cols=feature_cols,
        embed_dim=16,
        hidden_dims=[256, 128, 64],
        dropout=0.2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练
    print("\n开始训练...")
    best_val_auc = 0.0
    best_epoch = 0
    
    for epoch in range(1, TrainConfig.epochs + 1):
        start_time = time.time()
        
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:2d}/{TrainConfig.epochs}: "
              f"Loss={train_loss:.4f}, Train AUC={train_auc:.4f}, "
              f"Val AUC={val_metrics['auc']:.4f}, Val ECE={val_metrics['ece']:.4f}, "
              f"Time={epoch_time:.1f}s")
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), project_root / 'checkpoints' / 'exp01_baseline_best.pth')
    
    print(f"\n最佳验证 AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    
    # 在测试集上评估
    print("\n在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(project_root / 'checkpoints' / 'exp01_baseline_best.pth'))
    test_metrics, test_preds, test_labels, test_logits = evaluate(model, test_loader, device)
    
    print_metrics(test_metrics, prefix="测试集 ")
    
    # 保存结果
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 转换为可 JSON 序列化的类型
    test_metrics_serializable = {k: float(v) for k, v in test_metrics.items()}
    
    with open(results_dir / 'exp01_baseline_metrics.json', 'w') as f:
        json.dump(test_metrics_serializable, f, indent=2)
    
    np.save(results_dir / 'exp01_baseline_preds.npy', test_preds)
    np.save(results_dir / 'exp01_baseline_labels.npy', test_labels)
    np.save(results_dir / 'exp01_baseline_logits.npy', test_logits)
    
    # 生成报告
    report = f"""# 实验 01: 基线模型校准性能评估

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 实验配置

- 模型: MLP (Embedding + MLP)
- Embedding 维度: 16
- 隐藏层: [256, 128, 64]
- 训练轮数: {TrainConfig.epochs}
- 批次大小: {TrainConfig.batch_size}
- 学习率: {TrainConfig.learning_rate}
- 设备: {device}

## 结果

### 测试集指标

| 指标 | 值 |
|------|-----|
| AUC | {test_metrics['auc']:.4f} |
| LogLoss | {test_metrics['logloss']:.4f} |
| ECE | {test_metrics['ece']:.4f} |
| MCE | {test_metrics['mce']:.4f} |
| PCOC | {test_metrics['pcoc']:.4f} |

### 分析

{'✅ 校准较好 (PCOC 接近 1.0)' if 0.9 < test_metrics['pcoc'] < 1.1 else '⚠️ 校准需要改进'}
"""
    
    with open(results_dir / 'exp01_baseline.md', 'w') as f:
        f.write(report)
    
    print(f"\n结果已保存到 {results_dir}")
    print("\n" + "="*60)
    print("实验 01 完成!")
    print("="*60)


if __name__ == '__main__':
    # 创建必要目录
    (project_root / 'checkpoints').mkdir(exist_ok=True)
    main()
