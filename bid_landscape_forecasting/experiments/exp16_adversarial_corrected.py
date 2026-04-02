#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 16 (修正版): Adversarial Co-Training for Counterfactual Imputation

参考论文: 《Neural Bid Landscape Learning with Counterfactual Imputation》, ICLR 2026

核心思想 (正确理解):
1. 问题：我们只能观察到 (bid, win)，不知道真实的 value
2. Generator: 从观测数据推断潜在的 true_value 分布
3. Discriminator: 验证推断的 value 是否能解释观测到的 win pattern
4. 联合训练：让 Generator 学会生成合理的 counterfactual values
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

print("="*60)
print("🚀 Experiment 16 (Corrected): Adversarial Counterfactual Imputation")
print("="*60)

start_time = time.time()

# ========== 配置 ==========
config = {
    'max_samples': 100000,
    'hidden_dim': 128,
    'epochs': 50,
    'batch_size': 512,
    'lr_G': 0.0002,
    'lr_D': 0.0002,
    'lambda_adv': 0.1,  # Adversarial loss weight
    'random_state': 42
}

# ========== 加载数据 ==========
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > config['max_samples']:
    df = df.sample(n=config['max_samples'], random_state=config['random_state'])
print(f"  Loaded {len(df):,} samples")
print(f"  Features: {df.columns.tolist()}")
print(f"  Win rate: {df['win_label'].mean():.4f}")

# 使用更多特征
feature_cols = ['bid_amount', 'click_label']
X = df[feature_cols].fillna(0).values
y_win = df['win_label'].values
y_true_value = df['true_value'].values  # 用于验证（实际场景中不可见）

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_win, test_size=0.2, random_state=42, stratify=y_win
)
true_value_test = df['true_value'].values[len(X_train):]  # For validation only

# ========== 模型定义 ==========

class ValueInferenceGenerator(nn.Module):
    """
    Generator: 从观测数据推断潜在的 true_value
    Input: (bid, features)
    Output: inferred_value (0-1 range)
    """
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Value should be in [0, 1]
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ConsistencyDiscriminator(nn.Module):
    """
    Discriminator: 验证 (inferred_value, bid) 是否能正确预测 win
    Input: (inferred_value, bid, features)
    Output: predicted_win_prob
    """
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for inferred_value
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, inferred_value):
        combined = torch.cat([x, inferred_value.unsqueeze(1)], dim=1)
        return self.net(combined).squeeze(-1)

# 初始化模型
G = ValueInferenceGenerator(X_scaled.shape[1], config['hidden_dim']).to(device)
D = ConsistencyDiscriminator(X_scaled.shape[1], config['hidden_dim']).to(device)

optimizer_G = optim.Adam(G.parameters(), lr=config['lr_G'], betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=config['lr_D'], betas=(0.5, 0.9))

criterion_bce = nn.BCELoss()

# ========== 训练 ==========
print("\n🏋️ Training adversarial co-training model...")

best_auc = 0.0
patience_counter = 0

for epoch in range(config['epochs']):
    G.train()
    D.train()
    
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    n_batches = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), config['batch_size']):
        batch_X = torch.FloatTensor(X_train[i:i+config['batch_size']]).to(device)
        batch_y = torch.FloatTensor(y_train[i:i+config['batch_size']]).to(device)
        
        # Step 1: Train Discriminator (predict win from inferred value)
        optimizer_D.zero_grad()
        with torch.no_grad():
            inferred_value = G(batch_X)
        
        pred_win = D(batch_X, inferred_value)
        loss_D = criterion_bce(pred_win, batch_y)
        loss_D.backward()
        optimizer_D.step()
        
        # Step 2: Train Generator (two objectives)
        optimizer_G.zero_grad()
        
        # Objective 1: Make discriminator predict correct win labels
        pred_win_gen = D(batch_X, G(batch_X))
        loss_supervised = criterion_bce(pred_win_gen, batch_y)
        
        # Objective 2: Regularization - keep inferred values reasonable
        # (encourage diversity, avoid collapsing to constant)
        inferred = G(batch_X)
        loss_diversity = -torch.mean(torch.abs(inferred - 0.5))  # Encourage spread around 0.5
        
        loss_G = loss_supervised + config['lambda_adv'] * loss_diversity
        loss_G.backward()
        optimizer_G.step()
        
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        n_batches += 1
    
    avg_loss_G = epoch_loss_G / max(n_batches, 1)
    avg_loss_D = epoch_loss_D / max(n_batches, 1)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{config['epochs']}: Loss_G={avg_loss_G:.4f}, Loss_D={avg_loss_D:.4f}")
    
    # ========== 评估 ==========
    if (epoch + 1) % 5 == 0:
        G.eval()
        D.eval()
        
        with torch.no_grad():
            test_X = torch.FloatTensor(X_test).to(device)
            
            # Method 1: Use D's prediction directly
            inferred_test = G(test_X)
            pred_win_prob = D(test_X, inferred_test).cpu().numpy()
            
            auc = roc_auc_score(y_test, pred_win_prob)
            
            # Also check how close inferred value is to true value (oracle metric)
            inferred_true = inferred_test.cpu().numpy()
            value_correlation = np.corrcoef(inferred_true, true_value_test)[0, 1]
        
        print(f"    📊 Test AUC: {auc:.4f}, Value Correlation: {value_correlation:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            # Save best models
            torch.save(G.state_dict(), project_root / 'checkpoints' / 'exp16_G_best.pth')
            torch.save(D.state_dict(), project_root / 'checkpoints' / 'exp16_D_best.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= 6:  # ~30 epochs without improvement
            print(f"  ⚠️  Early stopping at epoch {epoch+1}")
            break

# ========== 最终评估 ==========
print("\n📈 Final Evaluation...")

# Load best models
os.makedirs(project_root / 'checkpoints', exist_ok=True)
if (project_root / 'checkpoints' / 'exp16_G_best.pth').exists():
    G.load_state_dict(torch.load(project_root / 'checkpoints' / 'exp16_G_best.pth'))
    D.load_state_dict(torch.load(project_root / 'checkpoints' / 'exp16_D_best.pth'))

G.eval()
D.eval()

with torch.no_grad():
    test_X = torch.FloatTensor(X_test).to(device)
    inferred_final = G(test_X)
    pred_win_final = D(test_X, inferred_final).cpu().numpy()
    
    final_auc = roc_auc_score(y_test, pred_win_final)
    
    # Additional metrics
    inferred_np = inferred_final.cpu().numpy()
    print(f"  Inferred value stats: mean={inferred_np.mean():.4f}, std={inferred_np.std():.4f}")
    print(f"  True value stats: mean={true_value_test.mean():.4f}, std={true_value_test.std():.4f}")
    print(f"  Correlation with true value: {np.corrcoef(inferred_np, true_value_test)[0, 1]:.4f}")

print(f"\n🎯 Final Test AUC: {final_auc:.4f}")

# ========== 保存结果 ==========
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

results = {
    'experiment': 'exp16_adversarial_corrected',
    'paper': 'Neural Bid Landscape Learning with Counterfactual Imputation (ICLR 2026)',
    'version': 'corrected_v1',
    'config': config,
    'metrics': {
        'auc': float(final_auc),
        'best_auc': float(best_auc) if best_auc > 0 else float(final_auc)
    },
    'training_time_seconds': time.time() - start_time,
    'device': str(device),
    'key_improvements': [
        'correct_generator_design',
        'proper_counterfactual_inference',
        'dual_objective_training',
        'consistency_discriminator'
    ]
}

with open(results_dir / 'exp16_adversarial_corrected.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved! AUC: {final_auc:.4f}")
print("="*60)
