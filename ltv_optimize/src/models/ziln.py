#!/usr/bin/env python3
"""
ZILN (Zero-Inflated Log-Normal) Model for LTV Prediction

Google, 2019: "Predicting Player Lifetime Value with Zero-Inflated Log-Normal"
Paper: https://arxiv.org/pdf/1912.07753

核心思想:
- 使用零膨胀对数正态分布拟合 LTV
- DNN 输出三个参数：π (zero prob), μ, σ (lognormal params)
- 损失函数：Negative Log-Likelihood
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class ZILNModel(nn.Module):
    """Zero-Inflated Log-Normal Model"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super(ZILNModel, self).__init__()
        
        # Build shared DNN
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*layers)
        
        # Three output heads
        # π: zero inflation probability (sigmoid -> [0, 1])
        self.pi_head = nn.Linear(prev_dim, 1)
        
        # μ: mean of log-normal (linear -> (-inf, +inf))
        self.mu_head = nn.Linear(prev_dim, 1)
        
        # σ: std of log-normal (softplus -> (0, +inf))
        self.sigma_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        """
        Forward pass
        Returns: pi, mu, sigma
        """
        h = self.shared_network(x)
        
        pi = torch.sigmoid(self.pi_head(h))  # [0, 1]
        mu = self.mu_head(h)  # (-inf, +inf)
        sigma = nn.functional.softplus(self.sigma_head(h)) + 1e-6  # (0, +inf), add epsilon for stability
        
        return pi, mu, sigma
    
    def predict(self, x):
        """
        Predict expected LTV value
        E[Y] = (1 - π) * exp(μ + σ²/2)
        """
        pi, mu, sigma = self.forward(x)
        expected_value = (1 - pi) * torch.exp(mu + 0.5 * sigma * sigma)
        return expected_value.detach().cpu().numpy()
    
    def predict_proba(self, x):
        """
        Predict probability of being a payer (1 - π)
        """
        pi, _, _ = self.forward(x)
        return (1 - pi).detach().cpu().numpy()


def ziln_negative_log_likelihood(pi, mu, sigma, y, eps=1e-10):
    """
    Compute Negative Log-Likelihood for Zero-Inflated Log-Normal distribution
    
    P(y) = π * δ₀(y) + (1-π) * LogNormal(y|μ, σ)
    
    For y = 0: log(P(y)) = log(π)
    For y > 0: log(P(y)) = log(1-π) + log(f_LogNormal(y|μ, σ))
    
    where f_LogNormal(y|μ, σ) = (1/(y*σ*sqrt(2π))) * exp(-(log(y)-μ)²/(2σ²))
    """
    # Mask for zeros and non-zeros
    is_zero = (y == 0).float()
    is_nonzero = (y > 0).float()
    
    # Log-likelihood for zeros: log(π)
    ll_zero = torch.log(pi.squeeze() + eps)
    
    # Log-likelihood for non-zeros: log(1-π) + log(f_LogNormal)
    log_y = torch.log(y + eps)
    log_pdf = -torch.log(sigma * torch.sqrt(2 * torch.pi)) - ((log_y - mu) ** 2) / (2 * sigma * sigma) - log_y
    ll_nonzero = torch.log(1 - pi.squeeze() + eps) + log_pdf
    
    # Combined log-likelihood
    ll = is_zero * ll_zero + is_nonzero * ll_nonzero
    
    # Negative log-likelihood (to minimize)
    nll = -ll.mean()
    
    return nll


def train_ziln(X_train, y_train, X_val, y_val, 
               epochs=1, batch_size=256, lr=1e-3, device='cuda'):
    """
    Train ZILN model
    """
    print(f"Training on device: {device}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = ZILNModel(input_dim).to(device)
    
    criterion = ziln_negative_log_likelihood
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_nll = float('inf')
    
    # Training loop (only 1 epoch per research_notes.md)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pi, mu, sigma = model(batch_X)
            loss = criterion(pi, mu, sigma, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_pi, all_mu, all_sigma, all_y = [], [], [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pi, mu, sigma = model(batch_X)
                loss = criterion(pi, mu, sigma, batch_y)
                
                total_val_loss += loss.item()
                all_pi.append(pi.cpu())
                all_mu.append(mu.cpu())
                all_sigma.append(sigma.cpu())
                all_y.append(batch_y.cpu())
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train NLL: {avg_train_loss:.4f}, Val NLL: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_nll:
            best_val_nll = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'val_nll': avg_val_loss
            }, Path('/mnt/workspace/open_research/autoresearch/ltv_optimize/results/exp002_ziln/best_model.pt'))
    
    return model, scaler, best_val_nll


def evaluate_ziln(model, scaler, X_val, y_val, device='cuda'):
    """
    Evaluate ZILN model with comprehensive metrics
    """
    model.eval()
    model.to(device)
    
    # Normalize
    X_val_scaled = scaler.transform(X_val)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    
    with torch.no_grad():
        # Get predictions
        pi, mu, sigma = model(X_val_tensor)
        
        # Expected LTV: E[Y] = (1 - π) * exp(μ + σ²/2)
        y_pred = (1 - pi) * torch.exp(mu + 0.5 * sigma * sigma)
        
        # Probability of being payer: 1 - π
        prob_payer = 1 - pi
        
        # Convert to numpy
        y_pred = y_pred.detach().cpu().numpy()
        prob_payer = prob_payer.detach().cpu().numpy()
    
    # Calculate metrics
    metrics = {}
    
    # AUC (payer prediction)
    y_binary = (y_val > 0).astype(int)
    if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
        metrics['auc'] = roc_auc_score(y_binary, prob_payer)
    
    # PCOC@K
    for k in [0.1, 0.2, 0.3]:
        top_k_idx = np.argsort(y_pred)[-int(len(y_pred) * k):]
        top_k_actual_ltv = y_val[top_k_idx].sum()
        total_ltv = y_val.sum()
        metrics[f'pcoc_{int(k*100)}'] = top_k_actual_ltv / total_ltv if total_ltv > 0 else 0
    
    # Regression metrics
    metrics['rmse'] = mean_squared_error(y_val, y_pred, squared=True) ** 0.5
    metrics['mae'] = mean_absolute_error(y_val, y_pred)
    
    # On payers only
    payer_mask = y_val > 0
    if payer_mask.sum() > 0:
        metrics['rmse_payers'] = mean_squared_error(y_val[payer_mask], y_pred[payer_mask], squared=True) ** 0.5
        metrics['mae_payers'] = mean_absolute_error(y_val[payer_mask], y_pred[payer_mask])
    
    # Log-Likelihood
    pi_tensor = torch.FloatTensor(pi.cpu().numpy())
    mu_tensor = torch.FloatTensor(mu.cpu().numpy())
    sigma_tensor = torch.FloatTensor(sigma.cpu().numpy())
    y_tensor = torch.FloatTensor(y_val)
    
    nll = ziln_negative_log_likelihood(pi_tensor, mu_tensor, sigma_tensor, y_tensor)
    metrics['test_nll'] = nll.item()
    
    return metrics


def run_ziln_experiment(data_path='/mnt/workspace/open_research/autoresearch/ltv_optimize/data/train_data.parquet'):
    """
    Run complete ZILN experiment
    """
    print("="*80)
    print("Experiment 002: ZILN (Zero-Inflated Log-Normal)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Split data
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    
    # Prepare features and labels
    feature_cols = [col for col in df_train.columns if col not in ['user_id', 'ltv_value']]
    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train['ltv_value'].values.astype(np.float32)
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val['ltv_value'].values.astype(np.float32)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train model
    print("\nTraining ZILN model...")
    model, scaler, best_val_nll = train_ziln(
        X_train, y_train, X_val, y_val,
        epochs=1,  # Follow research_notes.md: 1 epoch for streaming data
        batch_size=256,
        lr=1e-3,
        device=device
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_ziln(model, scaler, X_val, y_val, device)
    metrics['model'] = 'ZILN'
    metrics['epochs'] = 1
    
    # Print results
    print("\n" + "="*80)
    print("ZILN Results")
    print("="*80)
    print(f"AUC: {metrics.get('auc', 'N/A'):.4f}")
    print(f"PCOC@10: {metrics.get('pcoc_10', 'N/A'):.4f}")
    print(f"PCOC@20: {metrics.get('pcoc_20', 'N/A'):.4f}")
    print(f"PCOC@30: {metrics.get('pcoc_30', 'N/A'):.4f}")
    print(f"RMSE: {metrics.get('rmse', 'N/A'):.4f}")
    print(f"MAE: {metrics.get('mae', 'N/A'):.4f}")
    print(f"Test NLL: {metrics.get('test_nll', 'N/A'):.4f}")
    
    # Save results
    output_dir = Path('/mnt/workspace/open_research/autoresearch/ltv_optimize/results/exp002_ziln')
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return metrics


if __name__ == "__main__":
    run_ziln_experiment()
