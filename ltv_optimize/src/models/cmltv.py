#!/usr/bin/env python3
"""CMLTV: Contrastive Multi-view LTV Prediction"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from pathlib import Path
import json


class CMLTVModel(nn.Module):
    """CMLTV: Contrastive Learning + Heterogeneous Ensemble"""
    
    def __init__(self, input_dim):
        super().__init__()
        
        # Two views with dropout augmentation
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Heterogeneous regressors
        self.reg1 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())  # Standard
        self.reg2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))  # Log-transformed (no activation)
        self.reg3 = nn.Sequential(nn.Linear(64, 8), nn.Sigmoid())  # Binned classification style
    
    def forward(self, x):
        z1 = self.encoder1(x)
        z2 = self.encoder2(x)
        
        # Contrastive loss will be computed separately
        return z1, z2
    
    def predict(self, x):
        z1, z2 = self.forward(x)
        z = (z1 + z2) / 2  # Average representations
        
        # Ensemble predictions
        p1 = self.reg1(z).squeeze()
        p2 = torch.exp(self.reg2(z)).squeeze()  # Exp for log-transformed
        p3 = self.reg3(z).squeeze() * 20  # Scale up binned prediction
        
        # Simple average ensemble
        return (p1 + p2 + p3) / 3
    
    def contrastive_loss(self, z1, z2, temperature=0.5):
        """InfoNCE contrastive loss"""
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        similarity = torch.sum(z1 * z2, dim=1) / temperature
        return -similarity.mean()


def run_cmltv_experiment(data_path):
    """Run CMLTV experiment"""
    print("="*80)
    print("Experiment 005: CMLTV")
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
    
    model = CMLTVModel(X_train.shape[1]).to(device)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_scaled),
        torch.FloatTensor(y_train)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1):
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            z1, z2 = model(bx)
            pred = model.predict(bx).squeeze()
            
            # Combined loss: regression + contrastive
            reg_loss = mse_criterion(pred, by)
            con_loss = model.contrastive_loss(z1, z2)
            loss = reg_loss + 0.1 * con_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    metrics = {'model': 'CMLTV'}
    y_binary = (y_val > 0).astype(int)
    if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
        metrics['auc'] = roc_auc_score(y_binary, y_pred + np.random.randn(len(y_pred))*0.01)
    
    for k in [0.1, 0.2, 0.3]:
        top_k_idx = np.argsort(y_pred)[-int(len(y_pred)*k):]
        metrics[f'pcoc_{int(k*100)}'] = y_val[top_k_idx].sum() / y_val.sum()
    
    metrics['rmse'] = mean_squared_error(y_val, y_pred, squared=True)**0.5
    metrics['mae'] = np.mean(np.abs(y_val - y_pred))
    
    print(f"\nAUC: {metrics['auc']:.4f}, PCOC@10: {metrics['pcoc_10']:.4f}, RMSE: {metrics['rmse']:.4f}")
    
    output_dir = Path('/mnt/workspace/open_research/autoresearch/ltv_optimize/results/exp005_cmltv')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    run_cmltv_experiment('/mnt/workspace/open_research/autoresearch/ltv_optimize/data/train_data.parquet')
