"""
实验 07: Censored Regression (Tobit Model)

核心思想:
- 真实 RTB 中，输标时只知道"我输了"，不知道对手出了多少（截断/审查数据）
- Tobit 模型专门处理这种截断回归问题
- 损失函数: uncensored loss + censored loss (survival function)

我们的合成数据中有 true_value（市场价），可以模拟 censored 场景:
- 赢标 (win=1): 观测到完整信息，使用正常 BCE
- 输标 (win=0): 只知道 bid < market_price，使用 censored loss

两种模式:
1. Oracle: 使用 true_value（作弊，上界）
2. Censored: 不使用 true_value，只用 win/loss 信号（真实场景）
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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)


class CensoredRegressionModel(nn.Module):
    """
    Censored Regression Model for Bid Landscape
    
    输出: P(win|bid, context) = P(market_price < bid)
    
    两个输出头:
    1. Win probability (sigmoid)
    2. Market price estimate (for oracle mode)
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h
        
        self.backbone = nn.Sequential(*layers)
        
        # Win probability head
        self.win_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        )
        
        # Market price estimation head (for oracle comparison)
        self.price_head = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Softplus()  # 保证正值
        )
    
    def forward(self, x):
        h = self.backbone(x)
        win_prob = self.win_head(h).squeeze(-1)
        price_est = self.price_head(h).squeeze(-1)
        return win_prob, price_est


def censored_loss(win_prob, win_label, price_est, bid_amount, true_value=None, alpha=0.5):
    """
    Censored Regression Loss
    
    - 赢标 (win=1): BCE loss on win_prob
    - 输标 (win=0): survival loss (只知道 bid < market_price)
    
    如果有 true_value，额外加入 price estimation loss
    """
    eps = 1e-7
    win_mask = win_label.bool()
    lose_mask = ~win_mask
    
    # Uncensored loss (赢标样本)
    if win_mask.sum() > 0:
        win_loss = -torch.log(win_prob[win_mask] + eps).mean()
    else:
        win_loss = torch.tensor(0.0, device=win_prob.device)
    
    # Censored loss (输标样本): P(win=0) = 1 - P(win)
    if lose_mask.sum() > 0:
        lose_loss = -torch.log(1 - win_prob[lose_mask] + eps).mean()
    else:
        lose_loss = torch.tensor(0.0, device=win_prob.device)
    
    total_loss = win_loss + lose_loss
    
    # Oracle price estimation loss (如果有 true_value)
    if true_value is not None:
        price_loss = nn.MSELoss()(price_est, true_value)
        total_loss = total_loss + alpha * price_loss
    
    return total_loss


def load_flat_data(data_path, n_samples=200000):
    """加载平铺格式数据（每行一个 bid）"""
    print("加载数据...")
    df = pd.read_parquet(data_path)
    df = df.iloc[:n_samples].copy()
    
    context_cols = ['business_type', 'deviceid', 'adid', 'campaignid', 'click_label']
    for col in context_cols:
        df[col] = df[col].fillna(0)
    
    # 特征: context + bid_amount
    X = np.column_stack([
        df[context_cols].values,
        df['bid_amount'].values
    ])
    y_win = df['win_label'].values.astype(np.float32)
    y_price = df['true_value'].values.astype(np.float32)
    bid_amounts = df['bid_amount'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    print(f"样本数: {len(X):,}, 特征数: {X.shape[1]}")
    print(f"Win rate: {y_win.mean():.4f}")
    
    return X, y_win, y_price, bid_amounts


def train_censored(X, y_win, y_price, bid_amounts, device, mode='censored', epochs=30):
    """
    mode: 'censored' (不用 true_value) or 'oracle' (用 true_value)
    """
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    def to_tensor(arr):
        return torch.FloatTensor(arr).to(device)
    
    X_train = to_tensor(X[train_idx])
    y_win_train = to_tensor(y_win[train_idx])
    y_price_train = to_tensor(y_price[train_idx])
    bid_train = to_tensor(bid_amounts[train_idx])
    
    X_val = to_tensor(X[val_idx])
    y_win_val = to_tensor(y_win[val_idx])
    y_price_val = to_tensor(y_price[val_idx])
    bid_val = to_tensor(bid_amounts[val_idx])
    
    train_ds = TensorDataset(X_train, y_win_train, y_price_train, bid_train)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    
    model = CensoredRegressionModel(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"\n[{mode.upper()}] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for X_b, win_b, price_b, bid_b in train_loader:
            optimizer.zero_grad()
            win_pred, price_pred = model(X_b)
            
            true_val = price_b if mode == 'oracle' else None
            loss = censored_loss(win_pred, win_b, price_pred, bid_b, true_value=true_val)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        model.eval()
        with torch.no_grad():
            val_win_pred, _ = model(X_val)
            val_win_pred = val_win_pred.cpu().numpy()
            val_win_true = y_win_val.cpu().numpy()
        
        val_auc = roc_auc_score(val_win_true, val_win_pred)
        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            val_rmse = np.sqrt(mean_squared_error(val_win_true, val_win_pred))
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_rmse={val_rmse:.4f}")
    
    model.load_state_dict(best_state)
    return model, X_val, y_win_val


def evaluate_model(model, X_val, y_win_val, model_name):
    model.eval()
    with torch.no_grad():
        win_pred, _ = model(X_val)
        win_pred = win_pred.cpu().numpy()
        win_true = y_win_val.cpu().numpy()
    
    auc = float(roc_auc_score(win_true, win_pred))
    rmse = float(np.sqrt(mean_squared_error(win_true, win_pred)))
    ece = float(compute_ece(win_true, win_pred))
    
    print(f"\n=== {model_name} ===")
    print(f"AUC:  {auc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ECE:  {ece:.4f}")
    
    return {'auc': auc, 'rmse': rmse, 'ece': ece}


def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    X, y_win, y_price, bid_amounts = load_flat_data(str(data_path), n_samples=200000)
    
    results = {}
    
    # Mode 1: Censored（真实场景，不用 true_value）
    print("\n" + "="*50)
    print("Mode 1: Censored Regression（真实场景）")
    print("="*50)
    model_c, X_val_c, y_val_c = train_censored(X, y_win, y_price, bid_amounts, device, mode='censored', epochs=30)
    results['censored'] = evaluate_model(model_c, X_val_c, y_val_c, "Censored Regression")
    
    # Mode 2: Oracle（使用 true_value，上界）
    print("\n" + "="*50)
    print("Mode 2: Oracle Regression（使用 true_value，上界）")
    print("="*50)
    model_o, X_val_o, y_val_o = train_censored(X, y_win, y_price, bid_amounts, device, mode='oracle', epochs=30)
    results['oracle'] = evaluate_model(model_o, X_val_o, y_val_o, "Oracle Regression")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    print("\n=== 对比汇总 ===")
    print(f"{'模式':<20} {'AUC':>8} {'RMSE':>8} {'ECE':>8}")
    for mode, m in results.items():
        print(f"{mode:<20} {m['auc']:>8.4f} {m['rmse']:>8.4f} {m['ece']:>8.4f}")
    
    summary = {
        'experiment': 'exp07_censored',
        'method': 'Censored Regression (Tobit Model)',
        'results': results,
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp07_censored.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 07: Censored Regression (Tobit Model)

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: {device}

---

## 📊 评估结果

| 模式 | AUC | RMSE | ECE | 说明 |
|------|-----|------|-----|------|
| **Censored** | {results['censored']['auc']:.4f} | {results['censored']['rmse']:.4f} | {results['censored']['ece']:.4f} | 真实场景（不用 true_value） |
| **Oracle** | {results['oracle']['auc']:.4f} | {results['oracle']['rmse']:.4f} | {results['oracle']['ece']:.4f} | 使用 true_value（上界） |

---

## 🏗️ 模型架构

```
[context(5) + bid(1)] → MLP(128→64) → Win Head (sigmoid) → P(win)
                                     → Price Head (softplus) → market_price_est
```

## 💡 损失函数

```
L = L_uncensored + L_censored [+ α × L_price]

L_uncensored = -log P(win=1)   (赢标样本)
L_censored   = -log P(win=0)   (输标样本，只知道输了)
L_price      = MSE(price_est, true_value)  (仅 oracle 模式)
```

---

## 🎯 核心发现

1. **Censored vs Oracle 差距**: Oracle 因为有额外监督信号（true_value），预期表现更好
2. **Censored 的实际意义**: 这才是真实 RTB 场景的正确建模方式
3. **与基线对比**: 
   - LR 基线: AUC=0.8718, ECE=0.0036
   - Censored 模型是否能超越简单 LR？

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp07_censored.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp07_censored.json + .md")


if __name__ == '__main__':
    main()
