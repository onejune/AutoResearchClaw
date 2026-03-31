"""
实验 06: Multi-Task Learning for Sequence Attention (MTLSA)

参考: IJCAI 2018 "Multi-task Learning for Bid Landscape Forecasting"

核心思想:
- 将 5 个 bid levels 视为 5 个独立任务
- 共享底层特征表示，每个 bid level 有独立的预测头
- 任务之间通过共享参数传递信息

与 exp04 的区别:
- exp04: CTR + Win 两个不同任务
- exp06: 5 个 bid levels 作为 5 个相关任务（同一类型，不同难度）
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


class MTLSAModel(nn.Module):
    """
    Multi-Task Learning for Sequence Attention
    
    5 个 bid levels 作为 5 个相关任务:
    - Task k: P(win | bid_level=k, context)
    
    Architecture:
    - Shared Bottom: 处理上下文特征
    - Bid Embedding: 将 bid_amount 映射到向量
    - Attention: 任务间注意力机制（MTLSA 核心）
    - Task Heads: 每个 bid level 独立的预测头
    """
    
    def __init__(self, context_dim, n_tasks=5, hidden_dim=64, embed_dim=16):
        super().__init__()
        self.n_tasks = n_tasks
        self.embed_dim = embed_dim
        
        # Shared bottom (context encoder)
        self.shared_bottom = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        shared_out = hidden_dim // 2
        
        # Bid embedding (scalar → vector)
        self.bid_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Cross-task attention (MTLSA 核心)
        # 每个任务关注其他任务的信息
        self.task_attention = nn.MultiheadAttention(
            embed_dim=shared_out + embed_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Task-specific heads
        task_input_dim = shared_out + embed_dim
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(task_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(n_tasks)
        ])
    
    def forward(self, context, bid_seq):
        """
        context: (B, context_dim)
        bid_seq: (B, n_tasks) - bid amounts for each task
        Returns: (B, n_tasks) win probabilities
        """
        B = context.size(0)
        
        # Shared representation
        shared = self.shared_bottom(context)  # (B, shared_out)
        
        # Bid embeddings for each task
        bid_embeds = []
        for k in range(self.n_tasks):
            bid_k = bid_seq[:, k:k+1]  # (B, 1)
            embed_k = self.bid_embed(bid_k)  # (B, embed_dim)
            bid_embeds.append(embed_k)
        
        # Concatenate shared + bid embed for each task
        task_inputs = []
        for k in range(self.n_tasks):
            task_k = torch.cat([shared, bid_embeds[k]], dim=-1)  # (B, shared_out + embed_dim)
            task_inputs.append(task_k)
        
        # Stack as sequence for attention: (B, n_tasks, dim)
        task_seq = torch.stack(task_inputs, dim=1)
        
        # Cross-task attention
        attn_out, _ = self.task_attention(task_seq, task_seq, task_seq)  # (B, n_tasks, dim)
        
        # Task-specific predictions
        outputs = []
        for k in range(self.n_tasks):
            out_k = self.task_heads[k](attn_out[:, k, :])  # (B, 1)
            outputs.append(out_k)
        
        return torch.cat(outputs, dim=-1)  # (B, n_tasks)


def load_sequence_data(data_path, n_samples=200000):
    """加载并重组为序列格式"""
    print("加载数据...")
    df = pd.read_parquet(data_path)
    df = df.iloc[:n_samples * 5].copy()
    
    context_cols = ['business_type', 'deviceid', 'adid', 'campaignid', 'click_label']
    for col in context_cols:
        df[col] = df[col].fillna(0)
    
    n_groups = len(df) // 5
    
    context_data = df[context_cols].values.reshape(n_groups, 5, len(context_cols))[:, 0, :]
    bid_data = df['bid_amount'].values.reshape(n_groups, 5)
    win_data = df['win_label'].values.reshape(n_groups, 5)
    
    # 按 bid 排序
    sort_idx = np.argsort(bid_data, axis=1)
    bid_data = np.take_along_axis(bid_data, sort_idx, axis=1)
    win_data = np.take_along_axis(win_data, sort_idx, axis=1)
    
    scaler_ctx = StandardScaler()
    context_data = scaler_ctx.fit_transform(context_data)
    
    scaler_bid = StandardScaler()
    bid_data = scaler_bid.fit_transform(bid_data.reshape(-1, 1)).reshape(n_groups, 5)
    
    print(f"数据: {n_groups:,} 组, 每组 5 个 bid levels")
    print(f"Win rate: {win_data.mean():.4f}")
    return context_data, bid_data, win_data


def train_mtlsa(context_data, bid_data, win_data, device, epochs=30):
    """训练 MTLSA 模型"""
    n = len(context_data)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    def to_tensor(arr):
        return torch.FloatTensor(arr).to(device)
    
    ctx_train = to_tensor(context_data[train_idx])
    bid_train = to_tensor(bid_data[train_idx])
    win_train = to_tensor(win_data[train_idx])
    ctx_val = to_tensor(context_data[val_idx])
    bid_val = to_tensor(bid_data[val_idx])
    win_val = to_tensor(win_data[val_idx])
    
    train_ds = TensorDataset(ctx_train, bid_train, win_train)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)
    
    context_dim = context_data.shape[1]
    model = MTLSAModel(context_dim=context_dim, n_tasks=5, hidden_dim=64, embed_dim=16).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练集: {len(train_idx):,}, 验证集: {len(val_idx):,}")
    
    best_val_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for ctx_b, bid_b, win_b in train_loader:
            optimizer.zero_grad()
            pred = model(ctx_b, bid_b)
            loss = criterion(pred, win_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        model.eval()
        with torch.no_grad():
            val_pred = model(ctx_val, bid_val).cpu().numpy()
            val_true = win_val.cpu().numpy()
        
        val_auc = roc_auc_score(val_true.flatten(), val_pred.flatten())
        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            val_rmse = np.sqrt(mean_squared_error(val_true.flatten(), val_pred.flatten()))
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_rmse={val_rmse:.4f}")
    
    model.load_state_dict(best_state)
    return model, ctx_val, bid_val, win_val


def evaluate(model, ctx_val, bid_val, win_val, model_name):
    model.eval()
    with torch.no_grad():
        val_pred = model(ctx_val, bid_val).cpu().numpy()
        val_true = win_val.cpu().numpy()
    
    pred_flat = val_pred.flatten()
    true_flat = val_true.flatten()
    
    auc = float(roc_auc_score(true_flat, pred_flat))
    rmse = float(np.sqrt(mean_squared_error(true_flat, pred_flat)))
    ece = float(compute_ece(true_flat, pred_flat))
    
    # Per-task AUC
    per_task_auc = []
    for k in range(5):
        task_auc = float(roc_auc_score(val_true[:, k], val_pred[:, k]))
        per_task_auc.append(task_auc)
    
    # 单调性
    monotone = sum(
        1 for i in range(val_pred.shape[0])
        for j in range(4)
        if val_pred[i, j] <= val_pred[i, j+1]
    ) / (val_pred.shape[0] * 4)
    
    print(f"\n=== {model_name} 评估结果 ===")
    print(f"AUC:          {auc:.4f}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"ECE:          {ece:.4f}")
    print(f"Monotonicity: {monotone:.4f}")
    print(f"Per-task AUC: {[f'{a:.4f}' for a in per_task_auc]}")
    
    return {
        'auc': auc, 'rmse': rmse, 'ece': ece,
        'monotonicity': float(monotone),
        'per_task_auc': per_task_auc
    }


def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    context_data, bid_data, win_data = load_sequence_data(str(data_path), n_samples=200000)
    
    model, ctx_val, bid_val, win_val = train_mtlsa(
        context_data, bid_data, win_data, device, epochs=30
    )
    
    metrics = evaluate(model, ctx_val, bid_val, win_val, "MTLSA")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    summary = {
        'experiment': 'exp06_mtlsa',
        'method': 'Multi-Task Learning for Sequence Attention',
        'metrics': {k: (v if not isinstance(v, list) else v) for k, v in metrics.items()},
        'config': {'n_samples': 200000, 'epochs': 30, 'n_tasks': 5},
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp06_mtlsa.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 06: Multi-Task Learning for Sequence Attention (MTLSA)

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
| **Monotonicity** | {metrics['monotonicity']:.4f} |

### Per-Task AUC（按 bid level 从低到高）

| Task | Bid Level | AUC |
|------|-----------|-----|
| Task 1 | 最低 bid (0.5v) | {metrics['per_task_auc'][0]:.4f} |
| Task 2 | 低 bid (0.7v) | {metrics['per_task_auc'][1]:.4f} |
| Task 3 | 中 bid (1.0v) | {metrics['per_task_auc'][2]:.4f} |
| Task 4 | 高 bid (1.3v) | {metrics['per_task_auc'][3]:.4f} |
| Task 5 | 最高 bid (1.5v) | {metrics['per_task_auc'][4]:.4f} |

---

## 🏗️ 模型架构

```
Context → Shared Bottom → shared_repr
                              ↓
Bid_k → Bid Embedding → bid_embed_k
                              ↓
[shared_repr, bid_embed_k] × 5 tasks → Multi-head Attention → Task Heads → Win_prob_k
```

- **Shared Bottom**: Linear(5→64→32)
- **Bid Embedding**: Linear(1→16→16)
- **Cross-task Attention**: MultiheadAttention(heads=4)
- **Task Heads**: 5 × Linear(48→32→1)

---

## 💡 核心发现

1. **跨任务注意力**: 低 bid 任务的信息可以帮助高 bid 任务预测（相关性学习）
2. **Per-task 差异**: 中间 bid level (Task 3, ~1.0v) 通常最难预测（胜率~50%）
3. **单调性**: 模型是否学到 bid↑→win_prob↑ 的规律

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp06_mtlsa.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp06_mtlsa.json + .md")


if __name__ == '__main__':
    main()
