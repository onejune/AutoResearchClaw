#!/usr/bin/env python3
"""
ExpLTV: Expert-based LTV Prediction with Whale Detection

Tencent, 2023: "ExpLTV: A New Model for LTV Prediction with Expert Network"
Paper: https://arxiv.org/pdf/2308.12729

核心思想:
- 两阶段：先检测大 R 用户，再路由到专属专家
- Mixture-of-Experts 架构
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from pathlib import Path
import json


class ExpLTVModel(nn.Module):
    """ExpLTV: Mixture of Experts with Whale Detection"""
    
    def __init__(self, input_dim, num_experts=3, expert_hidden=[64, 32]):
        super().__init__()
        self.num_experts = num_experts
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Gating network (whale detection + expert selection)
        self.gating = nn.Linear(64, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, expert_hidden[0]),
                nn.ReLU(),
                nn.Linear(expert_hidden[0], expert_hidden[1]),
                nn.ReLU(),
                nn.Linear(expert_hidden[1], 1),
                nn.Softplus()
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        h = self.encoder(x)
        
        # Gating weights (softmax)
        gates = torch.softmax(self.gating(h), dim=1)
        
        # Expert predictions
        expert_preds = torch.stack([expert(h) for expert in self.experts], dim=1)
        
        # Weighted combination
        prediction = (gates * expert_preds).sum(dim=1)
        
        return prediction, gates
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            pred, _ = self.forward(x)
            return pred.detach().cpu().numpy().flatten()


def run_expltv_experiment(data_path):
    """Run ExpLTV experiment"""
    print("="*80)
    print("Experiment 004: ExpLTV")
    print("="*80)
    
    df = pd.read_parquet(data_path)
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    
    feature_cols = [c for c in df.columns if c not in ['user_id', 'ltv_value']]
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train['ltv_value'].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val['ltv_value'].values.astype(np.float32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    model = ExpLTVModel(X_train.shape[1], num_experts=3).to(device)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y_train)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred, _ = model(bx)
            loss = criterion(pred.squeeze(), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    metrics = {'model': 'ExpLTV'}
    y_binary = (y_val > 0).astype(int)
    if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
        metrics['auc'] = roc_auc_score(y_binary, y_pred + np.random.randn(len(y_pred))*0.01)
    
    for k in [0.1, 0.2, 0.3]:
        top_k_idx = np.argsort(y_pred)[-int(len(y_pred)*k):]
        metrics[f'pcoc_{int(k*100)}'] = y_val[top_k_idx].sum() / y_val.sum()
    
    metrics['rmse'] = mean_squared_error(y_val, y_pred, squared=True)**0.5
    metrics['mae'] = np.mean(np.abs(y_val - y_pred))
    
    print(f"\nAUC: {metrics['auc']:.4f}, PCOC@10: {metrics['pcoc_10']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    # Save
    output_dir = Path('/mnt/workspace/open_research/autoresearch/ltv_optimize/results/exp004_expltv')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    run_expltv_experiment('/mnt/workspace/open_research/autoresearch/ltv_optimize/data/train_data.parquet')
