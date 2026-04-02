#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 15 (优化版): Counterfactual VAE with End-to-End Training

优化点:
1. VAE + 判别头端到端训练
2. 反事实数据增强
3. 联合损失：ELBO + 监督学习
"""

import os, sys, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path
warnings = __import__('warnings'); warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

print("="*60)
print("🚀 Experiment 15 (Optimized): Counterfactual VAE")
print("="*60)

start_time = time.time()

config = {
    'max_samples': 100000,
    'latent_dim': 32,
    'beta': 1.0,  # KL weight
    'lambda_cf': 0.5,  # Counterfactual weight
    'batch_size': 256,
    'epochs': 50,
    'lr': 0.001,
    'random_state': 42
}

# ========== 加载数据 ==========
print("\n📂 Loading data...")
df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train.parquet')
if len(df) > config['max_samples']:
    df = df.sample(n=config['max_samples'], random_state=config['random_state'])
print(f"  Loaded {len(df):,} samples")

X = df[['bid_amount', 'true_value', 'click_label']].fillna(0).values
y = df['win_label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========== VAE with Classifier Head ==========
class CounterfactualVAE(nn.Module):
    def __init__(self, input_dim=3, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        # Classification head (uses latent representation)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def classify(self, z):
        return self.classifier(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        pred_prob = self.classify(z)
        return reconstruction, pred_prob, mu, logvar

model = CounterfactualVAE(X_scaled.shape[1], config['latent_dim']).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# ========== 训练 ==========
print("\n🏋️ Training VAE with end-to-end classification...")

for epoch in range(config['epochs']):
    model.train()
    
    indices = np.random.permutation(len(X_train))
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_ce = 0
    
    for start_idx in range(0, len(X_train), config['batch_size']):
        end_idx = min(start_idx + config['batch_size'], len(X_train))
        batch_indices = indices[start_idx:end_idx]
        
        batch_X = torch.FloatTensor(X_train[batch_indices]).to(device)
        batch_y = torch.FloatTensor(y_train[batch_indices]).to(device)
        
        optimizer.zero_grad()
        reconstruction, pred_prob, mu, logvar = model(batch_X)
        
        # Reconstruction loss
        recon_loss = nn.MSELoss()(reconstruction, batch_X)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Classification loss
        ce_loss = nn.BCELoss()(pred_prob.squeeze(), batch_y)
        
        # Counterfactual augmentation loss
        cf_loss = torch.tensor(0.0).to(device)
        if config['lambda_cf'] > 0:
            # Generate counterfactual by adding noise to latent
            z_noisy = mu + torch.randn_like(mu) * 0.3
            cf_pred = model.classify(z_noisy)
            cf_loss = nn.BCELoss()(cf_pred.squeeze(), batch_y)
        
        # Total loss
        loss = recon_loss + config['beta'] * kl_loss + ce_loss + config['lambda_cf'] * cf_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_ce += ce_loss.item()
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{config['epochs']}: Loss={total_loss/len(indices):.4f}, "
              f"Recon={total_recon/len(indices):.4f}, KL={total_kl/len(indices):.4f}, "
              f"CE={total_ce/len(indices):.4f}")

# ========== 评估 ==========
print("\n📈 Evaluating...")
model.eval()

test_X = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    _, pred_probs, _, _ = model(test_X)
    pred_probs = pred_probs.cpu().numpy()

auc = roc_auc_score(y_test, pred_probs)
print(f"  🎯 Test AUC: {auc:.4f}")

# Test counterfactual generation diversity
print("\n🔍 Testing counterfactual diversity...")
with torch.no_grad():
    mu, logvar = model.encode(test_X[:100])
    z_original = model.reparameterize(mu, logvar)
    z_cf = mu + torch.randn_like(mu) * 0.5
    diversity = (z_cf - z_original).abs().mean().item()
    print(f"  Latent space diversity: {diversity:.4f}")

# ========== 保存结果 ==========
print("\n💾 Saving results...")
results_dir = project_root / 'results'
os.makedirs(results_dir, exist_ok=True)

results = {
    'experiment': 'exp15_counterfactual_optimized',
    'paper': 'Neural Bid Landscape Learning with Counterfactual Imputation (ICLR 2026)',
    'config': config,
    'metrics': {
        'auc': auc,
        'latent_diversity': diversity,
        'vae_trained': True
    },
    'training_time_seconds': time.time() - start_time,
    'device': str(device),
    'total_params': total_params,
    'improvements': ['end_to_end_training', 'classifier_head', 'cf_augmentation', 'joint_loss']
}

with open(results_dir / 'exp15_counterfactual_optimized.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved! AUC: {auc:.4f}")
print("="*60)
