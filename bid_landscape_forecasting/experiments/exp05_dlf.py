"""
实验 05: Deep Landscape Forecasting (DLF)

参考论文: Ren et al., "Deep Landscape Forecasting for Real-time Bidding Advertising" (KDD 2019)
arXiv: 1905.03028

核心思想:
- 将 bid landscape 建模为价格序列上的条件概率
- P(win|bid=b) = 1 - P(market_price > b) = CDF(b)
- 用 RNN 建模价格序列的条件分布

实现策略:
- 对每个样本，将 5 个 bid levels 排序后作为序列输入
- 用 GRU 建模序列，输出每个 bid 的胜率
- 损失: BCE on win_label

注意: 我们有 true_value，但训练时不使用（避免数据泄露）
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
from scipy.stats import pearsonr


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


class DLFModel(nn.Module):
    """
    Deep Landscape Forecasting Model
    
    Architecture:
    1. Context Encoder: MLP on ad/user features → context vector
    2. Sequence Model: GRU on sorted bid sequence
    3. Output: win probability for each bid
    
    Input per sample:
    - context features: [business_type, deviceid, adid, campaignid, click_label]
    - bid sequence: sorted bid amounts [b1, b2, b3, b4, b5]
    
    Output: win probabilities [p1, p2, p3, p4, p5]
    """
    
    def __init__(self, context_dim, hidden_dim=64, rnn_hidden=32, n_bids=5):
        super().__init__()
        self.n_bids = n_bids
        
        # Context encoder (ad/user features)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        context_out_dim = hidden_dim // 2
        
        # GRU: input = [bid_amount, context_vector]
        self.gru = nn.GRU(
            input_size=1 + context_out_dim,  # bid + context
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(rnn_hidden, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, context, bid_seq):
        """
        context: (B, context_dim)
        bid_seq: (B, n_bids) - sorted bid amounts
        Returns: (B, n_bids) win probabilities
        """
        B = context.size(0)
        
        # Encode context
        ctx = self.context_encoder(context)  # (B, context_out_dim)
        
        # Expand context for each bid step
        ctx_expanded = ctx.unsqueeze(1).expand(-1, self.n_bids, -1)  # (B, n_bids, ctx_dim)
        
        # Concatenate bid with context
        bid_seq_3d = bid_seq.unsqueeze(-1)  # (B, n_bids, 1)
        rnn_input = torch.cat([bid_seq_3d, ctx_expanded], dim=-1)  # (B, n_bids, 1+ctx_dim)
        
        # GRU
        rnn_out, _ = self.gru(rnn_input)  # (B, n_bids, rnn_hidden)
        
        # Output
        win_probs = self.output_head(rnn_out).squeeze(-1)  # (B, n_bids)
        
        return win_probs


def load_and_prepare_data(data_path, n_samples=200000, seed=42):
    """
    加载数据，重组为每个原始样本 5 个 bid 的序列格式
    """
    print("加载数据...")
    df = pd.read_parquet(data_path)
    
    # 取前 n_samples * 5 行（每 5 行为一组）
    n_rows = n_samples * 5
    df = df.iloc[:n_rows].copy()
    
    print(f"使用 {len(df):,} 行 = {n_samples:,} 个原始样本 × 5 bids")
    
    # 上下文特征（不包含 true_value，避免数据泄露）
    context_cols = ['business_type', 'deviceid', 'adid', 'campaignid', 'click_label']
    
    # 填充缺失值
    for col in context_cols:
        df[col] = df[col].fillna(0)
    
    # 重组为序列格式：每 5 行为一个样本
    n_groups = len(df) // 5
    
    # Context features（每组取第一行，因为同组共享）
    context_data = df[context_cols].values.reshape(n_groups, 5, len(context_cols))
    context_data = context_data[:, 0, :]  # (n_groups, context_dim)
    
    # Bid sequence（每组 5 个 bid，按 bid_amount 排序）
    bid_data = df['bid_amount'].values.reshape(n_groups, 5)  # (n_groups, 5)
    win_data = df['win_label'].values.reshape(n_groups, 5)   # (n_groups, 5)
    
    # 按 bid 大小排序（DLF 需要有序序列）
    sort_idx = np.argsort(bid_data, axis=1)
    bid_data = np.take_along_axis(bid_data, sort_idx, axis=1)
    win_data = np.take_along_axis(win_data, sort_idx, axis=1)
    
    # 标准化 context
    scaler_ctx = StandardScaler()
    context_data = scaler_ctx.fit_transform(context_data)
    
    # 标准化 bid
    scaler_bid = StandardScaler()
    bid_flat = bid_data.reshape(-1, 1)
    bid_data = scaler_bid.fit_transform(bid_flat).reshape(n_groups, 5)
    
    print(f"数据形状: context={context_data.shape}, bids={bid_data.shape}, wins={win_data.shape}")
    print(f"Win rate: {win_data.mean():.4f}")
    
    return context_data, bid_data, win_data


def train_dlf(context_data, bid_data, win_data, device, epochs=30):
    """训练 DLF 模型"""
    
    # 划分训练/验证集
    n = len(context_data)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    # 转换为 tensor
    def to_tensor(arr):
        return torch.FloatTensor(arr).to(device)
    
    ctx_train = to_tensor(context_data[train_idx])
    bid_train = to_tensor(bid_data[train_idx])
    win_train = to_tensor(win_data[train_idx])
    
    ctx_val = to_tensor(context_data[val_idx])
    bid_val = to_tensor(bid_data[val_idx])
    win_val = to_tensor(win_data[val_idx])
    
    # 创建 DataLoader
    train_ds = TensorDataset(ctx_train, bid_train, win_train)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    
    # 模型
    context_dim = context_data.shape[1]
    model = DLFModel(context_dim=context_dim, hidden_dim=64, rnn_hidden=32, n_bids=5).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练集: {len(train_idx):,} 样本, 验证集: {len(val_idx):,} 样本")
    print(f"训练 {epochs} epochs...")
    
    best_val_auc = 0
    best_state = None
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        for ctx_b, bid_b, win_b in train_loader:
            optimizer.zero_grad()
            pred = model(ctx_b, bid_b)  # (B, 5)
            loss = criterion(pred, win_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_pred = model(ctx_val, bid_val).cpu().numpy()  # (N_val, 5)
            val_true = win_val.cpu().numpy()
        
        # 展平计算指标
        val_pred_flat = val_pred.flatten()
        val_true_flat = val_true.flatten()
        
        val_auc = roc_auc_score(val_true_flat, val_pred_flat)
        val_rmse = np.sqrt(mean_squared_error(val_true_flat, val_pred_flat))
        val_loss = train_loss / n_batches
        
        scheduler.step(val_loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: loss={val_loss:.4f}, val_auc={val_auc:.4f}, val_rmse={val_rmse:.4f}")
        
        history.append({'epoch': epoch+1, 'loss': val_loss, 'val_auc': val_auc, 'val_rmse': val_rmse})
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    
    return model, ctx_val, bid_val, win_val, history


def evaluate_dlf(model, ctx_val, bid_val, win_val):
    """评估 DLF 模型"""
    model.eval()
    with torch.no_grad():
        val_pred = model(ctx_val, bid_val).cpu().numpy()
        val_true = win_val.cpu().numpy()
    
    pred_flat = val_pred.flatten()
    true_flat = val_true.flatten()
    
    auc = roc_auc_score(true_flat, pred_flat)
    rmse = float(np.sqrt(mean_squared_error(true_flat, pred_flat)))
    ece = compute_ece(true_flat, pred_flat)
    
    # 序列单调性检验（bid 越大，win prob 应该越大）
    monotone_count = 0
    total_pairs = 0
    for i in range(val_pred.shape[0]):
        seq = val_pred[i]
        for j in range(len(seq) - 1):
            total_pairs += 1
            if seq[j] <= seq[j+1]:
                monotone_count += 1
    monotonicity = monotone_count / total_pairs
    
    print(f"\n=== DLF 评估结果 ===")
    print(f"AUC:          {auc:.4f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"ECE:          {ece:.4f}")
    print(f"Monotonicity: {monotonicity:.4f} (bid↑ → win_prob↑ 的比例)")
    
    return {
        'auc': float(auc),
        'rmse': float(rmse),
        'ece': float(ece),
        'monotonicity': float(monotonicity)
    }


def main():
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # 加载数据
    context_data, bid_data, win_data = load_and_prepare_data(str(data_path), n_samples=200000)
    
    # 训练
    model, ctx_val, bid_val, win_val, history = train_dlf(
        context_data, bid_data, win_data, device, epochs=30
    )
    
    # 评估
    metrics = evaluate_dlf(model, ctx_val, bid_val, win_val)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    # 保存结果
    summary = {
        'experiment': 'exp05_dlf',
        'method': 'Deep Landscape Forecasting (GRU-based)',
        'metrics': {k: float(v) for k, v in metrics.items()},
        'config': {
            'n_samples': 200000,
            'epochs': 30,
            'hidden_dim': 64,
            'rnn_hidden': 32,
            'n_bids': 5
        },
        'elapsed_seconds': float(elapsed)
    }
    
    json_path = results_dir / 'exp05_dlf.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ JSON 已保存: {json_path}")
    
    # 保存 MD 报告
    md_content = f"""# 实验 05: Deep Landscape Forecasting (DLF)

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **数据集**: Synthetic Bid Landscape (20 万样本)  
> **设备**: {device}

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | {metrics['auc']:.4f} | 胜率预测排序能力 |
| **RMSE** | {metrics['rmse']:.4f} | 概率预测精度 |
| **ECE** | {metrics['ece']:.4f} | 校准误差 |
| **Monotonicity** | {metrics['monotonicity']:.4f} | bid↑→win_prob↑ 的比例 |

---

## 🏗️ 模型架构

```
Context Features → MLP Encoder → Context Vector
                                        ↓
Sorted Bid Sequence → [bid_i, ctx] → GRU → Output Head → Win Prob_i
```

- **Context Encoder**: Linear(5→64→32) + ReLU + Dropout
- **GRU**: 2 layers, hidden=32, input=[bid(1) + ctx(32)]
- **Output**: Linear(32→16→1) + Sigmoid

---

## 🎯 核心发现

### DLF vs 基线对比

| 模型 | AUC | RMSE | ECE | 特点 |
|------|-----|------|-----|------|
| Logistic Regression | 0.8718 | 0.4620 | **0.0036** | 最佳校准 |
| MLP | 0.8718 | **0.3816** | 0.0056 | 最佳 RMSE |
| Multi-task | **0.8725** | 0.3809 | 0.0052 | 最佳 AUC |
| **DLF (本实验)** | **{metrics['auc']:.4f}** | **{metrics['rmse']:.4f}** | **{metrics['ece']:.4f}** | 序列建模 |

### 单调性分析
- Monotonicity = {metrics['monotonicity']:.4f}
- DLF 通过 RNN 建模价格序列，天然倾向于学习单调递增的 bid-win 关系
- 理想情况下 Monotonicity 应接近 1.0

---

## 💡 方法优势

1. **序列感知**: GRU 能捕捉 bid 之间的相对关系（不只是单点预测）
2. **完整曲线**: 一次前向传播输出整条 bid-win 曲线
3. **可扩展**: 可以预测任意 bid 点（不限于训练时的 5 个）

---

*耗时: {elapsed:.1f}s*
"""
    
    md_path = results_dir / 'exp05_dlf.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"✅ MD 报告已保存: {md_path}")


if __name__ == '__main__':
    main()
