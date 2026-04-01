"""
实验 14 (v2): Counterfactual Imputation - 改进版

问题分析:
- v1 版本 AUC 只有 0.26，效果不理想
- 原因：直接用回归模型预测 market price，然后计算 win rate，误差累积严重

改进方案:
1. 直接建模 P(win|bid, x) 而不是预测 market price
2. 用插补的 market price 作为辅助监督信号
3. Multi-task learning: 主任务 win prediction + 辅助任务 price prediction
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MultiTaskBidLandscape(nn.Module):
    """
    Multi-task Bid Landscape 模型
    
    主任务：P(win | bid, x) - 二分类
    辅助任务：market price prediction - 回归 (仅用于插补)
    """
    
    def __init__(self, input_dim, hidden=[128, 64]):
        super().__init__()
        
        # Shared backbone
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15)])
            prev = h
        
        self.backbone = nn.Sequential(*layers)
        
        # Task 1: Win probability (main)
        self.win_head = nn.Sequential(
            nn.Linear(prev, 1),
            nn.Sigmoid()
        )
        
        # Task 2: Market price prediction (auxiliary, for imputation)
        self.price_head = nn.Linear(prev, 1)
    
    def forward(self, x):
        h = self.backbone(x)
        win_prob = self.win_head(h).squeeze(-1)
        price_pred = self.price_head(h).squeeze(-1)
        return win_prob, price_pred


def counterfactual_imputation_v2(model, X, bids, events, config):
    """
    改进的反事实插补策略
    
    核心思想:
    - 对输标样本，找到一个价格 p 使得 P(win|p, x) = threshold
    - 例如：找到 p 使得 P(win|p, x) = 0.5，作为 market price 估计
    """
    device = next(model.parameters()).device
    X_t = torch.FloatTensor(X).to(device)
    bids_t = torch.FloatTensor(bids).to(device)
    
    model.eval()
    with torch.no_grad():
        win_probs, price_preds = model(X_t)
        win_probs = win_probs.cpu().numpy()
        price_preds = price_preds.cpu().numpy()
    
    imputed = np.zeros_like(bids)
    
    # 赢标样本：用真实值 (这里用 price_pred 作为代理)
    won_mask = (events == 1)
    imputed[won_mask] = price_preds[won_mask]
    
    # 输标样本：用反事实推理
    lost_mask = (events == 0)
    
    # 策略 1：直接用模型预测的价格
    imputed[lost_mask] = price_preds[lost_mask]
    
    # 策略 2：二分查找找到 P(win|p) = 0.5 的点
    # (简化版：假设线性关系，用当前 bid 和预测概率推算)
    for i in np.where(lost_mask)[0]:
        p_current = win_probs[i]
        bid_current = bids[i]
        
        # 如果 P(win|bid) < 0.5，说明 bid < market_price
        # 估算：market_price ≈ bid * (0.5 / p_current)
        if p_current > 0.01 and p_current < 0.5:
            estimated_price = bid_current * (0.5 / p_current)
            imputed[i] = max(imputed[i], estimated_price)
        else:
            # 保守估计：略高于当前 bid
            imputed[i] = max(imputed[i], bid_current * 1.1)
    
    # 物理约束：插补值 >= bid
    imputed = np.maximum(imputed, bids)
    
    return imputed


def train_multitask(X_train, bids_train, events_train, values_train,
                    X_val, bids_val, events_val, values_val,
                    config):
    """训练 multi-task 模型"""
    print("\n" + "="*60)
    print("Training Multi-task Model with Counterfactual Imputation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    X_tr = torch.FloatTensor(X_train).to(device)
    b_tr = torch.FloatTensor(bids_train).to(device)
    e_tr = torch.FloatTensor(events_train).to(device)
    v_tr = torch.FloatTensor(values_train).to(device)
    
    X_va = torch.FloatTensor(X_val).to(device)
    e_va = torch.FloatTensor(events_val).to(device)
    v_va = torch.FloatTensor(values_val).to(device)
    
    model = MultiTaskBidLandscape(
        X_train.shape[1],
        hidden=config.get('hidden', [128, 64])
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 30)
    
    dataset = TensorDataset(X_tr, b_tr, e_tr, v_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_win_loss = 0
        total_price_loss = 0
        
        for bx, bb, be, bv in loader:
            optimizer.zero_grad()
            
            win_probs, price_preds = model(bx)
            
            # Main task: Win prediction (BCE loss)
            win_loss = nn.BCELoss()(win_probs, be)
            
            # Auxiliary task: Price prediction (MSE, only for won samples or all with imputation)
            price_loss = ((price_preds - bv)**2).mean()
            
            # Total loss
            alpha = config.get('alpha', 0.7)  # win loss weight
            total_loss_batch = alpha * win_loss + (1 - alpha) * price_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_win_loss += win_loss.item()
            total_price_loss += price_loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_win_probs, _ = model(X_va)
            val_win_probs = val_win_probs.cpu().numpy()
            
            auc = roc_auc_score(events_val, val_win_probs)
            
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {total_loss/len(loader):.4f} | "
                  f"Win Loss: {total_win_loss/len(loader):.4f} | "
                  f"Price Loss: {total_price_loss/len(loader):.4f} | "
                  f"Val AUC: {auc:.4f}")
    
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    
    results = {
        'model': 'Multi-task + Counterfactual',
        'best_auc': best_auc,
        'config': config
    }
    
    print(f"\nBest Val AUC: {best_auc:.4f}")
    return model, results


def main():
    print("="*70)
    print("Experiment 14 (v2): Multi-task Counterfactual Imputation")
    print("="*70)
    
    config = {
        'lr': 0.001,
        'batch_size': 256,
        'epochs': 30,
        'hidden': [128, 64],
        'alpha': 0.8,  # 更重视 win prediction
        'max_iter': 3,
    }
    
    # Load data
    data_path = project_root / 'data' / 'bid_landscape_train_small.parquet'
    df = pd.read_parquet(data_path)
    
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)
    
    print(f"Samples: {len(df)}")
    
    # Features
    feature_cols = ['bid_amount', 'business_type', 'deviceid', 'adid']
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    
    # Format
    events = df['win_label'].values.astype(np.float32)
    true_values = df['true_value'].values.astype(np.float32)
    bids = df['bid_amount'].values.astype(np.float32)
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    value_min, value_max = true_values.min(), true_values.max()
    values_norm = (true_values - value_min) / (value_max - value_min + 1e-10)
    bids_norm = (bids - value_min) / (value_max - value_min + 1e-10)
    
    print(f"Win rate: {events.mean():.4f}")
    
    # Split
    X_train, X_val, b_train, b_val, e_train, e_val, v_train, v_val = \
        train_test_split(X, bids_norm, events, values_norm, 
                        test_size=0.2, random_state=42, stratify=events)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    all_results = []
    current_values = v_train.copy()
    
    # Iterative training
    for iteration in range(config['max_iter']):
        print(f"\n=== Iteration {iteration+1}/{config['max_iter']} ===")
        
        # Train
        model, results = train_multitask(
            X_train, b_train, e_train, current_values,
            X_val, b_val, e_val, v_val,
            config
        )
        
        # Update imputation
        if iteration < config['max_iter'] - 1:
            current_values = counterfactual_imputation_v2(
                model, X_train, b_train, e_train, config
            )
            print(f"Updated imputation mean (lost): {current_values[e_train==0].mean():.4f}")
        
        results['iteration'] = iteration + 1
        all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for r in all_results:
        print(f"Iteration {r['iteration']}: {r['model']} | AUC: {r['best_auc']:.4f}")
    
    # Save
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'exp14_counterfactual_v2.json', 'w') as f:
        json.dump({'results': all_results}, f, indent=2)
    
    md = "# Exp14 (v2): Multi-task Counterfactual Imputation\n\n"
    md += "| Iteration | Model | AUC |\n|-----------|-------|-----|\n"
    for r in all_results:
        md += f"| {r['iteration']} | {r['model']} | {r['best_auc']:.4f} |\n"
    
    with open(results_dir / 'exp14_counterfactual_v2.md', 'w') as f:
        f.write(md)
    
    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
