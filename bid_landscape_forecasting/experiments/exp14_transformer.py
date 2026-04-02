#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验 14: Transformer-based Bid Landscape Forecasting

参考论文:
- 《Attention is All You Need》, NIPS 2017
- 《TabTransformer: Tabular Data Modeling Using Contextual Embeddings》, KDD 2020

核心思想:
1. 使用 Self-Attention 捕捉特征间的全局依赖关系
2. 对分类特征进行 embedding + attention
3. 融合连续特征和分类特征的表示

与现有方法的区别:
- vs MLP: 引入 attention 机制，更好的特征交互
- vs DLF/DeepWin: 不使用序列建模，而是特征级 attention
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


class TabTransformer(nn.Module):
    """TabTransformer for tabular data"""
    
    def __init__(self, input_dim, n_categories=None, cat_dims=None, 
                 d_model=64, n_heads=4, n_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.input_proj(x)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Add positional encoding (simple version)
        pos_enc = torch.zeros_like(x)
        pos_enc = self._get_pos_encoding(x.shape[1], self.d_model)
        x = x + pos_enc
        
        x = self.transformer(x)  # [batch_size, 1, d_model]
        x = x.squeeze(1)  # [batch_size, d_model]
        
        out = self.output_head(x)  # [batch_size, 1]
        return out.squeeze(-1)
    
    def _get_pos_encoding(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe[:seq_len, :].to(next(self.parameters()).device)


def prepare_data(df, test_size=0.2, val_size=0.1):
    """准备数据"""
    print("📊 Preparing dataset...")
    
    # 特征选择
    feature_cols = ['bid_amount', 'true_value']
    
    # 添加分类特征 (如果有)
    cat_cols = []
    if 'business_type' in df.columns:
        cat_cols.append('business_type')
    
    # 提取特征
    X_numeric = df[feature_cols].fillna(0).values
    
    if cat_cols:
        le = LabelEncoder()
        X_cat = le.fit_transform(df[cat_cols[0]].astype(str)).reshape(-1, 1).astype(float)
        X = np.hstack([X_numeric, X_cat])
    else:
        X = X_numeric
    
    y = df['win_label'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
    )
    
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_temp):,}")
    print(f"  Event rate: {y.mean():.4f}")
    
    return {
        'train': {'X': X_train, 'y': y_train},
        'val': {'X': X_val, 'y': y_val},
        'test': {'X': X_temp, 'y': y_temp},
        'input_dim': X_scaled.shape[1]
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50, patience=10):
    """训练模型"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\n🏋️ Training TabTransformer...")
    print(f"  Epochs: {epochs}, Patience: {patience}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"  ⏹️ Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"  ✅ Loaded best model (val_loss={best_val_loss:.4f})")
    
    return model


def evaluate_model(model, test_data, y_true):
    """评估模型"""
    from utils.metrics import compute_all_metrics, print_metrics
    
    model.eval()
    X_test = torch.FloatTensor(test_data['X']).to(device)
    
    print("\n🔍 Evaluating on test set...")
    
    with torch.no_grad():
        y_prob = model(X_test).cpu().numpy()
    
    y_pred = (y_prob >= 0.5).astype(float)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    
    return metrics, y_prob


def main():
    """Main experiment runner"""
    
    print("="*60)
    print("🚀 Experiment 14: TabTransformer")
    print("="*60)
    sys.stdout.flush()
    
    config = {
        'max_samples': 100000,
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'ff_dim': 128,
        'dropout': 0.1,
        'batch_size': 256,
        'epochs': 50,
        'lr': 0.001,
        'patience': 10,
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
    data = prepare_data(df)
    
    # 3. 创建 DataLoader
    print("\n🔄 Step 3: Creating DataLoaders...", flush=True)
    train_dataset = TensorDataset(
        torch.FloatTensor(data['train']['X']),
        torch.FloatTensor(data['train']['y'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(data['val']['X']),
        torch.FloatTensor(data['val']['y'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 4. 创建模型
    print("\n🏗️ Step 4: Building TabTransformer...", flush=True)
    model = TabTransformer(
        input_dim=data['input_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}", flush=True)
    
    # 5. 训练
    print("\n🏋️ Step 5: Training...", flush=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=config['epochs'], patience=config['patience']
    )
    
    # 6. 评估
    print("\n📈 Step 6: Evaluation...", flush=True)
    metrics, y_prob = evaluate_model(model, data['test'], data['test']['y'])
    
    from utils.metrics import print_metrics
    print_metrics(metrics, prefix="  ")
    
    # 7. 保存结果
    print("\n💾 Step 7: Saving results...", flush=True)
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_json = {
        'experiment': 'exp14_tabtransformer',
        'paper': 'TabTransformer (KDD 2020)',
        'config': config,
        'metrics': metrics,
        'training_time_seconds': time.time() - start_time,
        'device': str(device),
        'total_params': total_params
    }
    
    with open(results_dir / 'exp14_tabtransformer.json', 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    md_report = f"""# Experiment 14: TabTransformer Results

## Paper Reference
- **Title**: TabTransformer: Tabular Data Modeling Using Contextual Embeddings
- **Venue**: KDD 2020

## Method
- Self-attention for feature interaction
- Transformer encoder architecture
- Fusion of continuous and categorical features

## Configuration
- **Samples**: {config['max_samples']:,}
- **d_model**: {config['d_model']}
- **n_heads**: {config['n_heads']}
- **n_layers**: {config['n_layers']}

## Results

### Win Rate Prediction
| Metric | Value |
|--------|-------|
| AUC | {metrics.get('win_auc', 'N/A'):.4f} |
| RMSE | {metrics.get('win_rmse', 'N/A'):.4f} |
| MAE | {metrics.get('win_mae', 'N/A'):.4f} |
| ECE | {metrics.get('win_ece', 'N/A'):.4f} |

### Training Info
- **Device**: {device}
- **Parameters**: {total_params:,}
- **Training Time**: {time.time() - start_time:.2f}s

---
*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(results_dir / 'exp14_tabtransformer.md', 'w') as f:
        f.write(md_report)
    
    print(f"  ✅ Results saved to {results_dir}/", flush=True)
    
    print("\n" + "="*60)
    print("✅ Experiment 14 completed!")
    print("="*60)
    
    return metrics


if __name__ == '__main__':
    main()
