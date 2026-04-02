#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 14: Inequality-Constrained Loss with Long-Tail Analysis

参考论文:
- 《Win Rate Estimation via Censored Data Modeling in RTB》, WWW 2025

核心思想:
1. 利用输标样本的边界信息：market_price > bid_amount
2. 设计不等式约束损失函数
3. 按流量分位进行分层评估（长尾分析）

与 exp13 的区别:
- exp13: DeepHit 生存分析框架
- exp14: 简化的不等式约束 + 详细的长尾分析
"""

import os
import sys
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


class BoundaryConstrainedMLP(nn.Module):
    """MLP with boundary-aware loss"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class InequalityConstrainedLoss(nn.Module):
    """
    Loss function with inequality constraints
    
    For losing samples (win_label=0):
    - We know market_price > bid_amount
    - This implies P(win|bid=market_price) should be higher than P(win|bid=current_bid)
    """
    
    def __init__(self, lambda_bound=0.5):
        super().__init__()
        self.lambda_bound = lambda_bound
    
    def forward(self, pred_probs, labels, bid_amounts, true_values, mask=None):
        """
        Args:
            pred_probs: [batch_size] - predicted win probability
            labels: [batch_size] - 1=win, 0=lose
            bid_amounts: [batch_size] - actual bid amounts
            true_values: [batch_size] - true value (proxy for market_price)
            mask: [batch_size] - optional mask for specific samples
        """
        # 1. Standard BCE loss
        bce_loss = nn.BCELoss()(pred_probs, labels)
        
        # 2. Boundary constraint loss
        # For losing samples: P(win|bid=true_value) should be > P(win|bid=bid_amount)
        # Approximation: use gradient to estimate P(win|bid=true_value)
        
        boundary_loss = torch.tensor(0.0).to(pred_probs.device)
        
        losing_mask = (labels == 0) & (true_values > bid_amounts)
        n_losing = losing_mask.sum()
        
        if n_losing > 0:
            # Simple constraint: predicted prob should decrease as bid decreases
            # For losing samples, bid < true_value, so we enforce monotonicity
            
            # Compute bid ratio
            bid_ratio = bid_amounts / (true_values + 1e-10)
            
            # Expected probability at true_value should be higher
            # Use a simple linear approximation
            expected_prob_at_true_value = pred_probs + 0.1 * (1 - bid_ratio)
            expected_prob_at_true_value = torch.clamp(expected_prob_at_true_value, 0, 1)
            
            # Constraint: the model should predict higher prob at true_value
            # Since we can't re-evaluate, use a proxy: penalize low gradients
            boundary_loss = F.mse_loss(
                pred_probs[losing_mask],
                expected_prob_at_true_value[losing_mask]
            )
        
        total_loss = bce_loss + self.lambda_bound * boundary_loss
        return total_loss


def prepare_long_tail_data(df, test_size=0.2, val_size=0.1):
    """准备数据并添加长尾分析标签"""
    print("📊 Preparing dataset with long-tail analysis...")
    
    # 特征选择
    feature_cols = ['bid_amount', 'true_value']
    X = df[feature_cols].fillna(0).values
    y = df['win_label'].values
    bid_amounts = df['bid_amount'].values
    true_values = df['true_value'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 数据集划分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    bid_train, bid_temp = train_test_split(bid_amounts, test_size=test_size, random_state=42)
    tv_train, tv_temp = train_test_split(true_values, test_size=test_size, random_state=42)
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
    )
    bid_train, bid_val = train_test_split(bid_train, test_size=val_ratio, random_state=42)
    tv_train, tv_val = train_test_split(tv_train, test_size=val_ratio, random_state=42)
    
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")
    print(f"  Event rate: {y.mean():.4f}")
    
    # 长尾分析：按 bid_amount 分位
    q_percentiles = [10, 25, 50, 75, 90]
    quantiles = np.percentile(bid_amounts, q_percentiles)
    print(f"  Bid amount quantiles: {dict(zip(q_percentiles, [f'{q:.3f}' for q in quantiles]))}")
    
    return {
        'train': {'X': X_train, 'y': y_train, 'bids': bid_train, 'tv': tv_train},
        'val': {'X': X_val, 'y': y_val, 'bids': bid_val, 'tv': tv_val},
        'test': {'X': X_temp, 'y': y_temp, 'bids': bid_temp, 'tv': tv_temp},
        'scaler': scaler,
        'input_dim': X_scaled.shape[1],
        'quantiles': dict(zip(q_percentiles, quantiles))
    }


def evaluate_with_long_tail_analysis(model, test_data, quantiles):
    """评估模型并进行详细的长尾分析"""
    model.eval()
    X_test = torch.FloatTensor(test_data['X']).to(device)
    y_true = test_data['y']
    bids = test_data['bids']
    
    print("\n🔍 Evaluating with long-tail analysis...")
    
    with torch.no_grad():
        y_prob = model(X_test).cpu().numpy()
    
    y_pred = (y_prob >= 0.5).astype(float)
    
    # Overall metrics
    overall_auc = roc_auc_score(y_true, y_pred)
    print(f"  Overall AUC: {overall_auc:.4f}")
    
    # Long-tail analysis by bid amount quantiles
    results = {'overall_auc': overall_auc}
    
    percentile_names = ['bottom_10%', 'q1_25%', 'median_50%', 'q3_75%', 'top_10%']
    thresholds = list(quantiles.values())
    
    print("\n  📊 Long-Tail Analysis by Bid Amount:")
    print("  " + "-"*60)
    
    for i, (pct, thresh) in enumerate(quantiles.items()):
        if i == 0:
            mask = bids <= thresh
        elif i == len(quantiles) - 1:
            mask = bids > thresholds[-2]
        else:
            mask = (bids > thresholds[i-1]) & (bids <= thresh)
        
        if mask.sum() > 10 and len(np.unique(y_true[mask])) > 1:
            segment_auc = roc_auc_score(y_true[mask], y_pred[mask])
            segment_acc = (y_pred[mask] == y_true[mask]).mean()
        else:
            segment_auc = np.nan
            segment_acc = np.nan
        
        results[f'bid_{pct}_auc'] = segment_auc
        results[f'bid_{pct}_count'] = int(mask.sum())
        
        auc_str = f"{segment_auc:.4f}" if not np.isnan(segment_auc) else "N/A"
        print(f"    {percentile_names[i]:12s}: AUC={auc_str}, Samples={mask.sum():,}")
    
    # Additional analysis: by win probability segments
    print("\n  📊 Analysis by Predicted Probability:")
    print("  " + "-"*60)
    
    prob_segments = [('very_low', 0, 0.2), ('low', 0.2, 0.4), ('medium', 0.4, 0.6), 
                     ('high', 0.6, 0.8), ('very_high', 0.8, 1.0)]
    
    for name, low, high in prob_segments:
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() > 10 and len(np.unique(y_true[mask])) > 1:
            segment_auc = roc_auc_score(y_true[mask], y_pred[mask])
            calibration = y_true[mask].mean() - y_prob[mask].mean()
        else:
            segment_auc = np.nan
            calibration = np.nan
        
        results[f'prob_{name}_auc'] = segment_auc
        results[f'prob_{name}_calibration'] = calibration
        results[f'prob_{name}_count'] = int(mask.sum())
        
        auc_str = f"{segment_auc:.4f}" if not np.isnan(segment_auc) else "N/A"
        cal_str = f"{calibration:.4f}" if not np.isnan(calibration) else "N/A"
        print(f"    {name:12s}: AUC={auc_str}, Calibration={cal_str}, Samples={mask.sum():,}")
    
    return results, y_prob


import torch.nn.functional as F

def main():
    """Main experiment runner"""
    
    print("="*60)
    print("🚀 Experiment 14: Inequality Constraints + Long-Tail Analysis")
    print("="*60)
    sys.stdout.flush()
    
    config = {
        'max_samples': 100000,
        'hidden_dims': [128, 64],
        'lambda_bound': 0.5,
        'batch_size': 256,
        'epochs': 30,
        'lr': 0.001,
        'patience': 8,
        'random_state': 42
    }
    
    start_time = time.time()
    
    # 1. 加载数据
    print("\n📂 Step 1: Loading data...", flush=True)
    df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
    print(f"  Loaded {len(df):,} samples", flush=True)
    if config['max_samples'] and len(df) > config['max_samples']:
        df = df.sample(n=config['max_samples'], random_state=config['random_state'])
        print(f"  Sampled to {len(df):,} samples", flush=True)
    
    # 2. 准备数据
    print("\n📊 Step 2: Preparing dataset...", flush=True)
    data = prepare_long_tail_data(df)
    
    # 3. 创建 DataLoader
    print("\n🔄 Step 3: Creating DataLoaders...", flush=True)
    train_dataset = TensorDataset(
        torch.FloatTensor(data['train']['X']),
        torch.FloatTensor(data['train']['y']),
        torch.FloatTensor(data['train']['bids']),
        torch.FloatTensor(data['train']['tv'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['val']['X']),
        torch.FloatTensor(data['val']['y']),
        torch.FloatTensor(data['val']['bids']),
        torch.FloatTensor(data['val']['tv'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 4. 创建模型
    print("\n🏗️ Step 4: Building model...", flush=True)
    model = BoundaryConstrainedMLP(
        input_dim=data['input_dim'],
        hidden_dims=config['hidden_dims']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}", flush=True)
    
    # 5. 训练
    print("\n🏋️ Step 5: Training...", flush=True)
    criterion = InequalityConstrainedLoss(lambda_bound=config['lambda_bound'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, batch_bids, batch_tv in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_bids = batch_bids.to(device)
            batch_tv = batch_tv.to(device)
            
            optimizer.zero_grad()
            pred_probs = model(batch_x)
            loss = criterion(pred_probs, batch_y, batch_bids, batch_tv)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_bids, batch_tv in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_bids = batch_bids.to(device)
                batch_tv = batch_tv.to(device)
                
                pred_probs = model(batch_x)
                loss = criterion(pred_probs, batch_y, batch_bids, batch_tv)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", flush=True)
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stopping at epoch {epoch+1}", flush=True)
            break
    
    # 6. 评估
    print("\n📈 Step 6: Evaluation...", flush=True)
    metrics, y_prob = evaluate_with_long_tail_analysis(model, data['test'], data['quantiles'])
    
    # 7. 保存结果
    print("\n💾 Step 7: Saving results...", flush=True)
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_json = {
        'experiment': 'exp14_boundary_constraints',
        'paper': 'Win Rate Estimation via Censored Data Modeling (WWW 2025)',
        'config': config,
        'metrics': metrics,
        'training_time_seconds': time.time() - start_time,
        'device': str(device),
        'total_params': total_params,
        'features': ['inequality_constraints', 'long_tail_analysis']
    }
    
    with open(results_dir / 'exp14_boundary.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"  ✅ Results saved!", flush=True)
    
    print("\n" + "="*60)
    print("✅ Experiment 14 completed!")
    print("="*60)
    
    return metrics


if __name__ == '__main__':
    main()
