#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 13: DeepHit - Discrete-Time Survival Analysis with Competing Risks

参考论文:
- 《Win Rate Estimation via Censored Data Modeling in RTB》, WWW 2025
- 《DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks》, AAAI 2018

核心思想:
1. 将连续时间离散化为多个时间区间 (bins)
2. 使用神经网络预测每个时间区间的风险概率 h(t|x)
3. 通过累积风险函数计算生存函数 S(t|x)
4. 使用负对数似然损失处理删失数据
5. **新增**: 边界信息利用 (输标时 market_price > bid)
6. **新增**: 长尾流量分层评估

与现有实验的区别:
- vs exp10 (Deep Cox PH): 
  - Cox PH: 比例风险假设，连续时间
  - DeepHit: 离散时间，更灵活的风险函数
- 新增边界信息利用：输标时 market_price > bid 的不等式约束
- 新增长尾流量分层评估
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


class DeepHitNetwork(nn.Module):
    """DeepHit: Deep Learning for Survival Analysis with Competing Risks"""
    
    def __init__(self, input_dim, n_time_bins=10, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        
        # Shared bottom
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_bottom = nn.Sequential(*layers)
        
        # Cause-specific heads (one for each time bin)
        self.time_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(n_time_bins)
        ])
        
        self.n_time_bins = n_time_bins
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        features = self.shared_bottom(x)  # [batch_size, hidden_dim]
        
        # Get risk probabilities for each time bin
        risk_probs = torch.stack([head(features).squeeze(-1) for head in self.time_heads], dim=1)
        # risk_probs: [batch_size, n_time_bins]
        
        # Normalize to sum to 1 (plus a "censored" probability)
        risk_probs = risk_probs / (risk_probs.sum(dim=1, keepdim=True) + 1e-10)
        
        return risk_probs


class DeepHitLossWithBounds(nn.Module):
    """DeepHit Loss with boundary information utilization"""
    
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # ranking loss weight
    
    def forward(self, risk_probs, times, events, boundaries=None):
        """
        Args:
            risk_probs: [batch_size, n_time_bins]
            times: [batch_size] - discrete time bins
            events: [batch_size] - 1=event observed, 0=censored
            boundaries: [batch_size] - optional boundary information
        """
        batch_size = risk_probs.size(0)
        
        # 1. Negative Log-Likelihood Loss
        log_risk_probs = torch.log(risk_probs + 1e-10)
        
        # For observed events: use the risk at the observed time
        # For censored: use the survival probability (cumprod of 1-risk)
        survival_probs = torch.cumprod(1 - risk_probs + 1e-10, dim=1)
        
        # NLL loss
        nll_event = -log_risk_probs[range(batch_size), times] * events
        nll_censor = -torch.log(survival_probs[:, -1] + 1e-10) * (1 - events)
        nll_loss = (nll_event + nll_censor).mean()
        
        # 2. Ranking Loss (for observed events)
        ranking_loss = torch.tensor(0.0).to(risk_probs.device)
        n_pairs = 0
        
        if events.sum() > 1:
            # Get indices of observed events
            event_indices = torch.where(events > 0)[0]
            
            for i_idx in event_indices:
                t_i = times[i_idx]
                
                # Find samples with longer survival times
                for j_idx in event_indices:
                    if i_idx != j_idx and times[j_idx] > t_i:
                        # Sample i should have higher cumulative risk than j at time t_i
                        cum_risk_i = risk_probs[i_idx, :t_i+1].sum()
                        cum_risk_j = risk_probs[j_idx, :t_i+1].sum()
                        
                        # Hinge loss: max(0, risk_j - risk_i + margin)
                        margin = 0.1
                        pair_loss = torch.relu(cum_risk_j - cum_risk_i + margin)
                        ranking_loss += pair_loss
                        n_pairs += 1
        
        if n_pairs > 0:
            ranking_loss = ranking_loss / n_pairs
        else:
            ranking_loss = torch.tensor(0.0).to(risk_probs.device)
        
        # 3. Boundary Loss (if available)
        boundary_loss = torch.tensor(0.0).to(risk_probs.device)
        if boundaries is not None and (boundaries > 0).sum() > 0:
            # For samples with boundary info, enforce that risk increases with bid
            mask = boundaries > 0
            if mask.sum() > 1:
                # Simple monotonicity constraint
                for i_idx in torch.where(mask)[0]:
                    for j_idx in torch.where(mask)[0]:
                        if i_idx != j_idx and boundaries[i_idx] > boundaries[j_idx]:
                            # Higher boundary should have higher cumulative risk
                            risk_i = risk_probs[i_idx].sum()
                            risk_j = risk_probs[j_idx].sum()
                            boundary_loss += torch.relu(risk_j - risk_i + 0.05)
                
                boundary_loss = boundary_loss / (mask.sum() * (mask.sum() - 1))
        
        total_loss = nll_loss + self.alpha * ranking_loss + 0.1 * boundary_loss
        return total_loss


def prepare_data_with_bounds(df, n_time_bins=10, test_size=0.2, val_size=0.1):
    """准备 DeepHit 格式的数据，包含边界信息"""
    print("📊 Preparing DeepHit dataset with boundary information...")
    
    # 离散化 bid_amount 为时间区间
    df = df.copy()
    df['time_bin'] = pd.qcut(df['bid_amount'], q=n_time_bins, labels=False, duplicates='drop')
    
    # 特征选择
    feature_cols = ['bid_amount', 'true_value']
    X = df[feature_cols].fillna(0).values
    
    # 标签：时间和事件
    times = df['time_bin'].values
    events = df['win_label'].values  # 1=赢标 (观察到事件), 0=输标 (删失)
    
    # 边界信息：对于输标的样本，market_price > bid_amount
    # 这里用 true_value 作为 market_price 的代理
    boundaries = np.where(events == 0, df['true_value'].values - df['bid_amount'].values, 0)
    boundaries = np.clip(boundaries, 0, None)  # 只保留正边界
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 数据集划分
    X_train, X_temp, times_train, times_temp, events_train, events_temp = train_test_split(
        X_scaled, times, events, test_size=test_size, random_state=42, stratify=events
    )
    boundaries_train, boundaries_temp = train_test_split(boundaries, test_size=test_size, random_state=42)
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, times_train, times_val, events_train, events_val = train_test_split(
        X_train, times_train, events_train, test_size=val_ratio, random_state=42, stratify=events_train
    )
    boundaries_train, boundaries_val = train_test_split(boundaries_train, test_size=val_ratio, random_state=42)
    
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")
    print(f"  Event rate: {events.mean():.4f}")
    print(f"  Boundary info available: {(boundaries > 0).mean():.2%}")
    
    # 长尾分析：按 bid_amount 分位
    q25, q75 = np.percentile(df['bid_amount'], [25, 75])
    print(f"  Bid amount range: [{df['bid_amount'].min():.2f}, {df['bid_amount'].max():.2f}]")
    print(f"  IQR: [{q25:.2f}, {q75:.2f}]")
    
    return {
        'train': {'X': X_train, 'times': times_train, 'events': events_train, 'boundaries': boundaries_train},
        'val': {'X': X_val, 'times': times_val, 'events': events_val, 'boundaries': boundaries_val},
        'test': {'X': X_temp, 'times': times_temp, 'events': events_temp, 'boundaries': boundaries_temp},
        'scaler': scaler,
        'input_dim': X_scaled.shape[1],
        'n_time_bins': n_time_bins,
        'bid_stats': {'q25': q25, 'q75': q75, 'min': df['bid_amount'].min(), 'max': df['bid_amount'].max()}
    }


def evaluate_model_with_tail_analysis(model, test_data, bid_stats):
    """评估模型并进行长尾分析"""
    model.eval()
    X_test = torch.FloatTensor(test_data['X']).to(device)
    y_true = test_data['events']
    
    print("\n🔍 Evaluating on test set...")
    
    with torch.no_grad():
        risk_probs = model(X_test)
        survival = torch.cumprod(1 - risk_probs + 1e-10, dim=1)
        win_prob = (1 - survival[:, -1]).cpu().numpy()
    
    y_pred = (win_prob >= 0.5).astype(float)
    
    # Basic metrics
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    
    # Long-tail analysis
    X_test_np = test_data['X']
    # Reverse transform to get bid_amount (first feature)
    X_original = model.cpu().shared_bottom[0].weight.data.numpy()  # This is approximate
    
    # Simple tail analysis based on predicted probabilities
    head_mask = win_prob < 0.3  # Low win probability (head)
    tail_mask = win_prob > 0.7  # High win probability (tail)
    
    results = {
        'overall_auc': auc,
        'head_auc': roc_auc_score(y_true[head_mask], y_pred[head_mask]) if head_mask.sum() > 10 and len(np.unique(y_true[head_mask])) > 1 else np.nan,
        'tail_auc': roc_auc_score(y_true[tail_mask], y_pred[tail_mask]) if tail_mask.sum() > 10 and len(np.unique(y_true[tail_mask])) > 1 else np.nan,
        'head_count': head_mask.sum(),
        'tail_count': tail_mask.sum()
    }
    
    print(f"  Overall AUC: {auc:.4f}")
    print(f"  Head (low prob) AUC: {results['head_auc']:.4f} ({results['head_count']} samples)")
    print(f"  Tail (high prob) AUC: {results['tail_auc']:.4f} ({results['tail_count']} samples)")
    
    return results, y_prob


def main():
    """Main experiment runner"""
    
    print("="*60)
    print("🚀 Experiment 13: DeepHit with Boundary Constraints")
    print("="*60)
    sys.stdout.flush()
    
    config = {
        'max_samples': 100000,
        'n_time_bins': 10,
        'hidden_dims': [128, 64],
        'dropout': 0.2,
        'batch_size': 256,
        'epochs': 50,
        'lr': 0.001,
        'patience': 10,
        'alpha': 0.5,
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
    data = prepare_data_with_bounds(df, n_time_bins=config['n_time_bins'])
    
    # 3. 创建 DataLoader
    print("\n🔄 Step 3: Creating DataLoaders...", flush=True)
    train_dataset = TensorDataset(
        torch.FloatTensor(data['train']['X']),
        torch.LongTensor(data['train']['times']),
        torch.FloatTensor(data['train']['events']),
        torch.FloatTensor(data['train']['boundaries'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['val']['X']),
        torch.LongTensor(data['val']['times']),
        torch.FloatTensor(data['val']['events']),
        torch.FloatTensor(data['val']['boundaries'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 4. 创建模型
    print("\n🏗️ Step 4: Building DeepHit model...", flush=True)
    model = DeepHitNetwork(
        input_dim=data['input_dim'],
        n_time_bins=config['n_time_bins'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}", flush=True)
    
    # 5. 训练
    print("\n🏋️ Step 5: Training...", flush=True)
    criterion = DeepHitLossWithBounds(alpha=config['alpha'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_times, batch_events, batch_boundaries in train_loader:
            batch_x = batch_x.to(device)
            batch_times = batch_times.to(device)
            batch_events = batch_events.to(device)
            batch_boundaries = batch_boundaries.to(device)
            
            optimizer.zero_grad()
            risk_probs = model(batch_x)
            loss = criterion(risk_probs, batch_times, batch_events, batch_boundaries)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_times, batch_events, batch_boundaries in val_loader:
                batch_x = batch_x.to(device)
                batch_times = batch_times.to(device)
                batch_events = batch_events.to(device)
                batch_boundaries = batch_boundaries.to(device)
                
                risk_probs = model(batch_x)
                loss = criterion(risk_probs, batch_times, batch_events, batch_boundaries)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}", flush=True)
        
        if patience_counter >= config['patience']:
            print(f"  ⏹️ Early stopping at epoch {epoch+1}", flush=True)
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"  ✅ Loaded best model (val_loss={best_val_loss:.4f})", flush=True)
    
    # 6. 评估
    print("\n📈 Step 6: Evaluation...", flush=True)
    metrics, y_prob = evaluate_model_with_tail_analysis(model, data['test'], data['bid_stats'])
    
    # 7. 保存结果
    print("\n💾 Step 7: Saving results...", flush=True)
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_json = {
        'experiment': 'exp13_deephit_final',
        'paper': 'Win Rate Estimation via Censored Data Modeling (WWW 2025)',
        'config': config,
        'metrics': metrics,
        'training_time_seconds': time.time() - start_time,
        'device': str(device),
        'total_params': total_params,
        'features': ['boundary_constraints', 'long_tail_analysis']
    }
    
    with open(results_dir / 'exp13_deephit.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    md_report = f"""# Experiment 13: DeepHit Final Results

## Paper Reference
- **Title**: Win Rate Estimation via Censored Data Modeling in RTB
- **Venue**: WWW 2025

## Method
- Discrete-time survival analysis
- Neural network for hazard function estimation
- **Boundary constraints**: Utilize market_price > bid for losing samples
- **Long-tail analysis**: Stratified evaluation by win probability

## Configuration
- **Samples**: {config['max_samples']:,}
- **Time Bins**: {config['n_time_bins']}
- **Hidden Dims**: {config['hidden_dims']}
- **Alpha (ranking loss)**: {config['alpha']}

## Results

### Overall Performance
| Metric | Value |
|--------|-------|
| AUC | {metrics.get('overall_auc', 'N/A'):.4f} |

### Long-Tail Analysis
| Segment | AUC | Samples |
|---------|-----|---------|
| Head (low prob) | {metrics.get('head_auc', 'N/A'):.4f if isinstance(metrics.get('head_auc'), float) else 'N/A'} | {metrics.get('head_count', 0):,} |
| Tail (high prob) | {metrics.get('tail_auc', 'N/A'):.4f if isinstance(metrics.get('tail_auc'), float) else 'N/A'} | {metrics.get('tail_count', 0):,} |

### Training Info
- **Device**: {device}
- **Parameters**: {total_params:,}
- **Training Time**: {time.time() - start_time:.2f}s

---
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(results_dir / 'exp13_deephit.md', 'w') as f:
        f.write(md_report)
    
    print(f"  ✅ Results saved to {results_dir}/", flush=True)
    
    print("\n" + "="*60)
    print("✅ Experiment 13 completed!")
    print("="*60)
    
    return metrics


if __name__ == '__main__':
    main()
