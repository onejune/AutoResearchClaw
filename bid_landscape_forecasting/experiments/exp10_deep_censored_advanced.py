"""
实验 10: Advanced Deep Censored Learning for Win Rate Estimation

核心思想:
- 使用 Deep Cox Proportional Hazards Model (DeepSurv)
- 或者 Deep Hit (结合Cox和Weibull模型的优点)
- 更高级的生存分析方法处理RTB删失数据

在RTB场景中:
- 赢标: 观测到事件发生 (win=1)
- 输标: 右删失数据 (win=0, 但实际市场价更高)
- 使用负偏心似然函数训练模型
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


class DeepCoxPH(nn.Module):
    """
    Deep Cox Proportional Hazards Model for RTB
    
    使用深度神经网络估计风险分数 (log partial hazard)
    在RTB中，风险分数可解释为 win likelihood
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 输出 log partial hazard (风险分数)
        self.hazard_score = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        h = self.backbone(x)
        log_hazard = self.hazard_score(h).squeeze(-1)
        return log_hazard


def cox_negative_log_likelihood(log_hazards, events, sort_idx):
    """
    Cox比例风险模型的负对数似然损失
    
    Args:
        log_hazards: 模型输出的风险分数
        events: 事件指示器 (1=观测到事件, 0=删失)
        sort_idx: 按时间/风险排序的索引
    """
    # 按风险分数降序排列 (高风险排前面)
    sorted_log_hazards = log_hazards[sort_idx]
    sorted_events = events[sort_idx]
    
    # 计算部分似然函数
    # 对于每个事件发生的个体，计算其相对于风险集的概率
    hazard_exps = torch.exp(sorted_log_hazards)
    
    # 构建风险集累积和
    cumsum_exps = torch.flip(torch.cumsum(torch.flip(hazard_exps, dims=[0]), dim=0), dims=[0])
    
    # 事件个体的 log hazard
    event_log_hazards = sorted_log_hazards[sorted_events.bool()]
    
    # 风险集的 log cumulative hazard
    event_cumsum_log = torch.log(cumsum_exps)[sorted_events.bool()]
    
    # 负对数似然
    neg_ll = torch.mean(event_cumsum_log - event_log_hazards)
    
    return neg_ll


def load_data_for_advanced_censored(data_path, n_samples=50000):
    """为高级删失学习加载数据"""
    print("加载数据用于高级删失学习...")
    df = pd.read_parquet(data_path)
    df = df.iloc[:n_samples].copy()  # 使用较少样本以加快训练
    
    context_cols = ['business_type', 'deviceid', 'adid', 'campaignid', 'click_label']
    for col in context_cols:
        df[col] = df[col].fillna(0)
    
    # 特征: context + bid_amount
    X = np.column_stack([
        df[context_cols].values,
        df['bid_amount'].values
    ])
    
    # 事件: win=1 (观测到), lose=0 (删失)
    events = df['win_label'].values.astype(np.float32)
    
    # 在RTB中，我们可以用bid_amount作为"时间"概念
    # 高出价更可能赢，所以bid_amount可以作为风险顺序的proxy
    times = df['bid_amount'].values.astype(np.float32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    
    print(f"样本数: {len(X):,}, 特征数: {X.shape[1]}")
    print(f"事件率 (win rate): {events.mean():.4f}")
    
    return X, events, times


def train_advanced_censored(X, events, times, device, epochs=20):
    """训练高级删失模型"""
    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    def to_tensor(arr):
        return torch.FloatTensor(arr).to(device)
    
    X_train = to_tensor(X[train_idx])
    events_train = to_tensor(events[train_idx])
    times_train = to_tensor(times[train_idx])
    X_val = to_tensor(X[val_idx])
    events_val = to_tensor(events[val_idx])
    times_val = to_tensor(times[val_idx])
    
    # 对训练数据按时间排序
    train_sort_idx = torch.argsort(times_train, descending=True)
    
    train_ds = TensorDataset(X_train, events_train, times_train)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)  # 不打乱，保持排序
    
    model = DeepCoxPH(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练集: {len(train_idx):,}, 验证集: {len(val_idx):,}")
    
    best_val_auc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for X_b, events_b, times_b in train_loader:
            optimizer.zero_grad()
            
            log_hazards = model(X_b)
            
            # 按时间排序（高bid在前）
            sort_idx = torch.argsort(times_b, descending=True)
            
            # Cox损失
            loss = cox_negative_log_likelihood(log_hazards, events_b, sort_idx)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_log_hazards = model(X_val)
            # 将log hazard转换为风险概率
            # 使用sigmoid归一化到[0,1]
            val_risk_scores = torch.sigmoid(val_log_hazards)
            val_auc = roc_auc_score(events_val.cpu().numpy(), val_risk_scores.cpu().numpy())
            avg_loss = total_loss / n_batches
            scheduler.step(avg_loss)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            val_rmse = np.sqrt(mean_squared_error(events_val.cpu().numpy(), val_risk_scores.cpu().numpy()))
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_auc={val_auc:.4f}, val_rmse={val_rmse:.4f}")
    
    model.load_state_dict(best_state)
    
    # 返回验证集上的预测结果
    model.eval()
    with torch.no_grad():
        val_log_hazards = model(X_val)
        val_risk_scores = torch.sigmoid(val_log_hazards)
    
    return model, val_risk_scores, events_val


def evaluate_advanced_censored(val_risk_scores, events_val):
    """评估高级删失模型"""
    val_risk = val_risk_scores.cpu().numpy()
    val_true = events_val.cpu().numpy()
    
    auc = float(roc_auc_score(val_true, val_risk))
    rmse = float(np.sqrt(mean_squared_error(val_true, val_risk)))
    ece = float(compute_ece(val_true, val_risk))
    
    print(f"\n=== Advanced Deep Censored Learning 评估 ===")
    print(f"AUC:  {auc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"ECE:  {ece:.4f}")
    
    return {
        'auc': auc, 'rmse': rmse, 'ece': ece
    }


def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    data_path = project_root / 'data' / 'bid_landscape_train.parquet'
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    X, events, times = load_data_for_advanced_censored(str(data_path), n_samples=50000)
    
    model, val_risk_scores, events_val = train_advanced_censored(
        X, events, times, device, epochs=20
    )
    
    metrics = evaluate_advanced_censored(val_risk_scores, events_val)
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed:.1f}s")
    
    summary = {
        'experiment': 'exp10_deep_censored_advanced',
        'method': 'Advanced Deep Censored Learning (Deep Cox PH) for Win Rate Estimation',
        'metrics': {k: float(v) for k, v in metrics.items()},
        'config': {
            'n_samples': 50000,
            'epochs': 20,
            'hidden_dims': [128, 64, 32],
            'device': str(device),
            'model_type': 'Deep Cox Proportional Hazards'
        },
        'elapsed_seconds': float(elapsed)
    }
    
    with open(results_dir / 'exp10_deep_censored_advanced.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    md_content = f"""# 实验 10: Advanced Deep Censored Learning for Win Rate Estimation

> **实验日期**: {time.strftime('%Y-%m-%d %H:%M')}  
> **方法**: Deep Cox Proportional Hazards Model
> **数据集**: Synthetic Bid Landscape (5 万样本)  
> **设备**: {device}

---

## 📊 评估结果

| 指标 | 值 | 说明 |
|------|----|------|
| **AUC** | {metrics['auc']:.4f} | 排序能力 |
| **RMSE** | {metrics['rmse']:.4f} | 概率预测精度 |
| **ECE** | {metrics['ece']:.4f} | 校准误差 |

---

## 🏗️ 模型架构

```
[context(5) + bid(1)] → MLP(128→64→32) → Risk Score (log partial hazard)
```

**核心创新**:
- 使用Cox比例风险模型处理删失数据
- 深度学习增强风险评分能力
- 适用于RTB中的右删失数据结构

---

## 💡 方法原理

在RTB环境中应用生存分析：
- **赢标** (event=1): 观测到"胜利事件"
- **输标** (event=0): 右删失，只知道"市场价更高"
- **风险评分**: 模型输出个体相对风险，高分意味着更可能赢标

**Cox回归优势**:
- 不假设基础生存分布
- 处理删失数据能力强
- 解释性强 (风险比)

---

## 🎯 与其他方法对比

| 模型 | AUC | RMSE | ECE | 特点 |
|------|-----|------|-----|------|
| LR Baseline | 0.8718 | 0.4620 | 0.0036 | 简单可靠 |
| MLP | 0.8718 | 0.3816 | 0.0056 | 深度学习 |
| Multi-task | 0.8725 | 0.3809 | 0.0052 | 联合优化 |
| Censored (Real) | 0.8674 | 0.3848 | 0.0040 | 经典删失回归 |
| Conformal | 0.8655 | 0.3863 | 0.0079 | 置信区间保证 |
| **Deep Cox PH** | **{metrics['auc']:.4f}** | **{metrics['rmse']:.4f}** | **{metrics['ece']:.4f}** | 生存分析框架 |

**高级删失学习的独特价值**:
- 基于坚实的生存分析理论
- 有效处理RTB中的删失数据
- 风险评分提供直观解释
- 适合序列化和在线学习

---

*耗时: {elapsed:.1f}s*
"""
    
    with open(results_dir / 'exp10_deep_censored_advanced.md', 'w') as f:
        f.write(md_content)
    
    print(f"✅ 结果已保存: results/exp10_deep_censored_advanced.json + .md")


if __name__ == '__main__':
    main()