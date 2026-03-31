"""
实验 08: Quantile Regression Neural Network

核心思想:
- 不预测单点概率，而是预测市场价格的分位数分布
- 有了分位数分布，就能推导出任意 bid 的胜率
- P(win|bid) = P(market_price < bid) = CDF(bid)

方法:
1. Quantile Regression NN: 预测多个分位数 τ ∈ {0.1, 0.2, ..., 0.9}
2. 从分位数推导 bid-win 曲线
3. 与直接预测 win probability 对比

优势:
- 输出完整的市场价格分布（不只是单点）
- 可以计算任意 bid 的胜率（插值）
- 天然支持不确定性量化

我们有 true_value，可以直接用分位数损失训练市场价格预测
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


def pinball_loss(pred, target, tau):
    """
    Pinball (Quantile) Loss
    L_τ(y, q) = τ * max(y-q, 0) + (1-τ) * max(q-y, 0)
    """
    diff = target - pred
    return torch.where(diff >= 0, tau * diff, (tau - 1) * diff).mean()


class QuantileRegressionNN(nn.Module):
    """
    Quantile Regression Neural Network
    
    输入: context features
    输出: 多个分位数 q_τ(market_price | context)
    
    从分位数推导胜率:
    P(win|bid) ≈ 插值 CDF(bid)
    """
    
    def __init__(self, input_dim, n_quantiles=9, hidden_dims=[128, 64]):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.quantiles = torch.linspace(0.1, 0.9, n_quantiles)
        
        # Shared backbone
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
        
        # Quantile output heads (一个头输出所有分位数)
        self.quantile_head = nn.Linear(prev_dim, n_quantiles)
        
        # Softplus 保证分位数单调递增（通过输出差分）
        # 实际上直接输出，训练时用 pinball loss 约束
    
    def forward(self, x):
        h = self.backbone(x)
        quantiles = self.quantile_head(h)  # (B, n_quantiles)
        # 确保单调性：用 softplus 累加
        # q_1 = softplus(out_1), q_2 = q_1 + softplus(out_2), ...
        q = torch.zeros_like(quantiles)
        q[:, 0] = torch.nn.functional.softplus(quantiles[:, 0])
        for i in range(1, self.n_quantiles):
            q[:, i] = q[:, i-1] + torch.nn.functional.softplus(quantiles[:, i])
        return q
    
    def predict_win_prob(self, x, bid_amounts):
        """
        从分位数预测胜率
        P(win|bid) = P(market_price < bid) ≈ CDF(bid)
        
        通过线性插值分位数 CDF
        """
        quantile_preds = self.forward(x).detach().cpu().numpy()  # (B, n_quantiles)
        bid_np = bid_amounts.cpu().numpy() if torch.is_tensor(bid_amounts) else bid_amounts
        
        quantile_levels = np.linspace(0.1, 0.9, self.n_quantiles)
        
        win_probs = np.zeros(len(bid_np))
        for i in range(len(bid_np)):
            bid = bid_np[i]
            q_vals = quantile_preds[i]
            
            # 线性插值
            if bid <= q_vals[0]:
                win_probs[i] = 0.1 * (bid / q_vals[0]) if q_vals[0] > 0 else 0.0
            elif bid >= q_vals[-1]:
                win_probs[i] = 0.9 + 0.1 * min((bid - q_vals[-1]) / (q_vals[-1] + 1e-7), 1.0)
            else:
                # 找到插值区间
                for j in range(len(q_vals) - 1):
                    if q_vals[j] <= bid <= q_vals[j+1]:
                        t = (bid - q_vals[j]) / (q_vals[j+1] - q_vals[j] + 1e-7)
                        win_probs[i] = quantile_levels[j] + t * (quantile_levels[j+1] - quantile_levels[j])
                        break
        
        return np.clip(win_probs, 0, 1)


def load_data(data_path, n_samples=200000):
    """加载数据"""
    print("加载数据...")
    df = pd.read_parquet(data_path)
    df = df.iloc[:n_samples].copy()
    
    context_cols = ['business_type', 'deviceid', 'adid', 'campaignid', 'click_label']
    for col in context_cols:
        df[col] = df[col].fillna(0)
    
    X = df[context_cols].values.astype(np.float32)
    y_price = df['true_value'].values.astype(np.float32)  # 目标：市场价格分位数
    y_win = df['win_label'].values.astype(np.float32)
    bid_amounts = df['bid_amount'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    print(f"样本数: {len(X):,}, Win rate: {y_win.mean():.4f}")
    print(f"Market price range: [{y_price.min():.4f}, {y_price.max():.4f}]")
    
    return X, y_price, y_win, bid_amounts


def train_quantile(X, y_price, device, epochs=30, n_quantiles=9):
    """训练分位数回归模型"""
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    def to_tensor(arr):
        return torch.FloatTensor(arr).to(device)
    
    X_train = to_tensor(X[train_idx])
    y_train = to_tensor(y_price[train_idx])
    X_val = to_tensor(X[val_idx])
    y_val = to_tensor(y_price[val_idx])
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    
    model = QuantileRegressionNN(input_dim=X.shape[1], n_quantiles=n_quantiles).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    quantile_levels = torch.linspace(0.1, 0.9, n_quantiles).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练 {epochs} epochs, {n_quantiles} 个分位数...")
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            q_pred = model(X_b)  # (B, n_quantiles)
            
            # 每个分位数的 pinball loss
            loss = sum(
                pinball_loss(q_pred[:, k], y_b, quantile_levels[k])
                for k in range(n_quantiles)
            ) / n_quantiles
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        model.eval()
        with torch.no_grad():
            q_val = model(X_val)
            val_loss = sum(
                pinball_loss(q_val[:, k], y_val, quantile_levels[k])
                for k in range(n_quantiles)
            ).item() / n_quantiles
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={total_loss/n_batches:.4f}, val_loss={val_loss:.4f}")
    
    model.load_state_dict(best_state)
    return model, val_idx


def evaluate_quantile(model, X, y_win, bid_amounts, val_idx, device):
    """评估：用分位数预测胜率，与真实 win_label 对比"""
    X_val = torch.FloatTensor(X[val_idx]).to(device)
    bid_val = bid_amounts[val_idx]
    win_val = y_win[val_idx]
    
    # 预测胜率
    win_pred = model.predict_win_prob(X_val, bid_val)
    
    auc = float(roc_auc_score(win_val, win_pred))
    rmse = float(np.sqrt(mean_squared_error(win_val, win_pred)))
    ece = float(compute_ece(win_val, win_pred))
    
    # 分位数覆盖率（校准检验）
    model.eval()
    with torch.no_grad():
        q_pred = model(X_val).cpu().numpy()
    
    y_price_val = None  # 我们没有在这里传入 y_price，用 bid 作为代理
    
    print(f"\n=== Quantile Regression NN 评估 ===")
    print(f"AUC:  {auc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ECE:  {ece:.4f}")
    print(f"\n分位数统计 (val set 均值):")
    quantile_levels = np.linspace(0.1, 0.9, 9)
    for k, tau in enumerate(quantile_levels):
        print(f"  Q{tau:.1f}: {q_pred[:, k].mean():.4f} ± {q_pred[:, k].std():.4f}")
    
    return {'auc': auc, 'rmse': rmse, 'ece': ece}


def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    X, y_price, y_win, bid_amounts = load_data(str(data_path), n_samples=200000)
    
    model, val_idx = train_quantile(X, y_price, device, epochs=30, n_quantiles=9)
    
    metrics = evaluate_quantile(model, X, y_win, bid_amounts, val_idx, device)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    summary = {
        'experiment': 'exp08_quantile',
        'method': 'Quantile Regression Neural Network',
        'metrics': {k: float(v) for k, v in metrics.items()},
        'config': {
            'n_samples': 200000,
            'epochs': 30,
            'n_quantiles': 9,
            'quantile_levels': list(np.linspace(0.1, 0.9, 9).round(1))
        },
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp08_quantile.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 08: Quantile Regression Neural Network

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: {device}

---

## 📊 评估结果

| 指标 | 值 |
|------|----|
| **AUC** | {metrics['auc']:.4f} |
| **RMSE** | {metrics['rmse']:.4f} |
| **ECE** | {metrics['ece']:.4f} |

---

## 🏗️ 模型架构

```
Context Features → MLP(128→64) → Quantile Head → [Q0.1, Q0.2, ..., Q0.9]
                                                         ↓
                                              线性插值 CDF(bid)
                                                         ↓
                                               P(win|bid) = CDF(bid)
```

**单调性保证**: 用 softplus 累加差分确保 Q0.1 ≤ Q0.2 ≤ ... ≤ Q0.9

---

## 💡 方法优势

1. **完整分布**: 输出市场价格的完整分位数分布
2. **任意 bid 胜率**: 通过插值 CDF 可以预测任意出价的胜率
3. **不确定性量化**: 分位数宽度反映预测不确定性
4. **无需 win_label 训练**: 只需要市场价格（true_value）即可训练

---

## 🎯 与其他方法对比

| 模型 | AUC | RMSE | ECE | 特点 |
|------|-----|------|-----|------|
| LR | 0.8718 | 0.4620 | 0.0036 | 最佳校准 |
| MLP | 0.8718 | 0.3816 | 0.0056 | 最佳 RMSE |
| Multi-task | 0.8725 | 0.3809 | 0.0052 | 最佳 AUC |
| **Quantile NN** | **{metrics['auc']:.4f}** | **{metrics['rmse']:.4f}** | **{metrics['ece']:.4f}** | 完整分布 |

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp08_quantile.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp08_quantile.json + .md")


if __name__ == '__main__':
    main()
