"""
实验 02: Deep Learning Models for Bid Landscape Forecasting

模型:
1. MLP (Multi-Layer Perceptron)
2. TabNet (Attentive Interpretable Tabular Learning)
3. FT-Transformer (Feature Transformation Transformer)
"""

import os, sys
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def prepare_data(df_train, df_val):
    """准备数据"""
    # 特征
    numeric_cols = ['bid_amount', 'true_value']
    categorical_cols = ['business_type']
    
    # 数值特征标准化
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(df_train[numeric_cols])
    X_val_num = scaler.transform(df_val[numeric_cols])
    
    # One-hot encode business_type
    bt_dummies_train = pd.get_dummies(df_train[categorical_cols], prefix='bt')
    bt_dummies_val = pd.get_dummies(df_val[categorical_cols], prefix='bt')
    
    # 对齐 columns
    bt_dummies_train, bt_dummies_val = bt_dummies_train.align(
        bt_dummies_val, join='left', axis=1, fill_value=0
    )
    
    X_train = np.hstack([X_train_num, bt_dummies_train.values])
    X_val = np.hstack([X_val_num, bt_dummies_val.values])
    
    y_train = df_train['win_label'].values
    y_val = df_val['win_label'].values
    
    # 转换为 tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    return X_train_t, y_train_t, X_val_t, y_val_t, X_train.shape[1]


def train_model(model, train_loader, val_X, val_y, epochs=50, lr=1e-3, device='cuda'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_X.to(device))
            val_auc = roc_auc_score(val_y.cpu(), val_outputs.cpu())
            val_loss = criterion(val_outputs, val_y.to(device)).item()
        
        scheduler.step(val_loss)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train Loss={np.mean(train_losses):.4f}, Val AUC={val_auc:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_auc


def run_mlp(X_train, y_train, X_val, y_val, input_dim, batch_size=1024, device='cuda'):
    """运行 MLP"""
    print("\n" + "="*60)
    print("1. MLP (Multi-Layer Perceptron)")
    print("="*60)
    
    # Create data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create and train model
    model = MLP(input_dim, hidden_dims=[256, 128, 64], dropout=0.3)
    model, best_auc = train_model(model, train_loader, X_val, y_val, epochs=50, device=device)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_prob = model(X_val.to(device)).cpu().numpy()
    
    metrics = {
        'auc': roc_auc_score(y_val, y_prob),
        'rmse': np.sqrt(mean_squared_error(y_val, y_prob)),
    }
    
    print(f"\nFinal Results:")
    print(f"  AUC:  {metrics['auc']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    
    return model, metrics


def compute_ece(y_true, y_prob, n_bins=10):
    """计算 ECE"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() / len(y_prob) * abs(bin_acc - bin_conf)
    
    return ece


def main():
    print("="*80)
    print("实验 02: Deep Learning Models for Bid Landscape Forecasting")
    print("="*80)
    
    # 加载数据
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    df = pd.read_parquet(data_path).head(500000)
    
    print(f"\nLoaded {len(df)} samples")
    
    # 划分训练/验证集
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['win_label'])
    print(f"Train: {len(df_train)}, Val: {len(df_val)}")
    
    # 准备数据
    X_train, y_train, X_val, y_val, input_dim = prepare_data(df_train, df_val)
    print(f"Input dimension: {input_dim}")
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 运行 MLP
    mlp_model, mlp_metrics = run_mlp(X_train, y_train, X_val, y_val, input_dim, device=device)
    
    # 计算 ECE
    mlp_model.eval()
    with torch.no_grad():
        y_prob = mlp_model(X_val.to(device)).cpu().numpy()
    
    mlp_metrics['ece'] = compute_ece(y_val, y_prob)
    print(f"  ECE:  {mlp_metrics['ece']:.4f}")
    
    # 保存结果
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'mlp': mlp_metrics
    }
    
    with open(results_dir / 'exp02_deep_learning.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = f"""# 实验 02: Deep Learning Models

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 方法对比

| Method | AUC | RMSE | ECE |
|--------|-----|------|-----|
| MLP | {mlp_metrics['auc']:.4f} | {mlp_metrics['rmse']:.4f} | {mlp_metrics['ece']:.4f} |

## 与基线对比

| Method | AUC | RMSE | ECE |
|--------|-----|------|-----|
| Logistic Regression (exp01) | 0.8718 | 0.4620 | 0.0036 |
| XGBoost (exp01) | 0.8714 | 0.4625 | 0.0050 |
| MLP (exp02) | {mlp_metrics['auc']:.4f} | {mlp_metrics['rmse']:.4f} | {mlp_metrics['ece']:.4f} |

## 核心发现

- **MLP vs LR**: {('略优' if mlp_metrics['auc'] > 0.8718 else '略差')} ({mlp_metrics['auc'] - 0.8718:+.4f})
- **校准效果**: {"良好" if mlp_metrics['ece'] < 0.01 else "需要改进"}
- **结论**: {"Deep learning 有优势" if mlp_metrics['auc'] > 0.8718 else "简单模型已足够"}

---

*实验报告 - 牛顿 🍎*
"""
    
    with open(results_dir / 'exp02_deep_learning.md', 'w') as f:
        f.write(report)
    
    print(f"\n✅ 结果已保存到 {results_dir}")


if __name__ == '__main__':
    main()
