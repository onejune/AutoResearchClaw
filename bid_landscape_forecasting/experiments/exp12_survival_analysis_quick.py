"""
实验 12 (快速版): Survival Analysis for Bid Landscape

简化的生存分析实现，用于快速验证
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


# ==================== 模型定义 ====================

class CoxPH(nn.Module):
    """Cox Proportional Hazards - 线性风险函数"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)
    
    def cox_loss(self, risk, times, events):
        # 按时间排序
        idx = torch.argsort(times, descending=True)
        risk_s = risk[idx]
        events_s = events[idx]
        
        exp_risk = torch.exp(risk_s)
        cumsum = torch.cumsum(exp_risk, dim=0)
        
        log_num = (risk_s * events_s).sum()
        log_denom = torch.log(cumsum + 1e-10).sum()
        
        return -(log_num - log_denom) / (events.sum() + 1e-10)


class DeepCox(nn.Module):
    """Deep Cox - 神经网络风险函数"""
    def __init__(self, input_dim, hidden=[32, 16]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.risk_head = nn.Linear(prev, 1)
    
    def forward(self, x):
        h = self.backbone(x)
        return self.risk_head(h).squeeze(-1)
    
    def cox_loss(self, risk, times, events):
        idx = torch.argsort(times, descending=True)
        risk_s = risk[idx]
        events_s = events[idx]
        
        exp_risk = torch.exp(risk_s)
        cumsum = torch.cumsum(exp_risk, dim=0)
        
        log_num = (risk_s * events_s).sum()
        log_denom = torch.log(cumsum + 1e-10).sum()
        
        return -(log_num - log_denom) / (events.sum() + 1e-10)


class DeepHit(nn.Module):
    """DeepHit - 离散时间生存模型"""
    def __init__(self, input_dim, K=30, hidden=[64, 32]):
        super().__init__()
        self.K = K
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.15)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.pmf_head = nn.Sequential(nn.Linear(prev, K), nn.Softmax(dim=-1))
    
    def forward(self, x):
        h = self.backbone(x)
        return self.pmf_head(h)
    
    def survival(self, pmf):
        # S(t) = P(T > t) = cumsum from right
        return torch.cumsum(torch.flip(pmf, [-1]), dim=-1).flip([-1])
    
    def loss(self, pmf, times, events, alpha=0.7):
        N, K = pmf.shape
        eps = 1e-10
        ll = 0.0
        
        for i in range(N):
            t_norm = times[i]
            bin_idx = min(int(t_norm * K), K-1)
            
            if events[i] == 1:
                ll += torch.log(pmf[i, bin_idx] + eps)
            else:
                if bin_idx < K-1:
                    surv = pmf[i, bin_idx+1:].sum() + eps
                else:
                    surv = torch.tensor(eps, device=pmf.device)
                ll += torch.log(surv)
        
        return -alpha * ll / N


# ==================== 主流程 ====================

def main():
    print("="*60)
    print("Exp12: Survival Analysis (Quick Test)")
    print("="*60)
    
    # 加载数据
    data_path = project_root / 'data' / 'bid_landscape_train_small.parquet'
    df = pd.read_parquet(data_path)
    
    # 采样加速
    if len(df) > 30000:
        df = df.sample(30000, random_state=42)
    
    print(f"Samples: {len(df)}")
    
    # 准备特征
    feature_cols = ['bid_amount', 'business_type', 'deviceid', 'adid']
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    
    # 生存分析格式
    events = df['win_label'].values.astype(np.float32)  # 1=赢标，0=删失
    times = np.where(events == 1, 
                     df['true_value'].values,
                     df['bid_amount'].values).astype(np.float32)
    
    # 归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    times = (times - times.min()) / (times.max() - times.min() + 1e-10)
    
    print(f"Win rate: {events.mean():.4f}, Censoring: {1-events.mean():.4f}")
    
    # 划分数据集
    X_train, X_val, t_train, t_val, e_train, e_val = \
        train_test_split(X, times, events, test_size=0.2, random_state=42, stratify=events)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 转换为 tensor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr = torch.FloatTensor(X_train).to(device)
    t_tr = torch.FloatTensor(t_train).to(device)
    e_tr = torch.FloatTensor(e_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    
    results = []
    
    # === 1. Cox PH ===
    print("\n[1/3] Training Cox PH...")
    model = CoxPH(X_train.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)
    
    dataset = TensorDataset(X_tr, t_tr, e_tr)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    
    best_auc = 0
    for epoch in range(15):
        model.train()
        for bx, bt, be in loader:
            opt.zero_grad()
            risk = model(bx)
            loss = model.cox_loss(risk, bt, be)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_va)).cpu().numpy()
            auc = roc_auc_score(e_val, probs)
            if auc > best_auc:
                best_auc = auc
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: AUC={auc:.4f}")
    
    print(f"  Best AUC: {best_auc:.4f}")
    results.append({'model': 'Cox PH', 'auc': best_auc})
    
    # === 2. Deep Cox ===
    print("\n[2/3] Training Deep Cox...")
    model = DeepCox(X_train.shape[1], [32, 16]).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    best_auc = 0
    for epoch in range(20):
        model.train()
        for bx, bt, be in loader:
            opt.zero_grad()
            risk = model(bx)
            loss = model.cox_loss(risk, bt, be)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_va)).cpu().numpy()
            auc = roc_auc_score(e_val, probs)
            if auc > best_auc:
                best_auc = auc
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: AUC={auc:.4f}")
    
    print(f"  Best AUC: {best_auc:.4f}")
    results.append({'model': 'Deep Cox', 'auc': best_auc})
    
    # === 3. DeepHit ===
    print("\n[3/3] Training DeepHit...")
    model = DeepHit(X_train.shape[1], K=30, hidden=[64, 32]).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    best_auc = 0
    for epoch in range(20):
        model.train()
        for bx, bt, be in loader:
            opt.zero_grad()
            pmf = model(bx)
            loss = model.loss(pmf, bt, be)
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            pmf = model(X_va)
            surv = model.survival(pmf)
            probs = surv[:, 15].cpu().numpy()  # 中位时间的生存概率
            probs = np.clip(probs, 0, 1)
            auc = roc_auc_score(e_val, probs)
            if auc > best_auc:
                best_auc = auc
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: AUC={auc:.4f}")
    
    print(f"  Best AUC: {best_auc:.4f}")
    results.append({'model': 'DeepHit', 'auc': best_auc})
    
    # 汇总
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['model']:15s} | AUC: {r['auc']:.4f}")
    
    # 保存结果
    out_dir = project_root / 'results'
    out_dir.mkdir(exist_ok=True)
    
    with open(out_dir / 'exp12_survival_quick.json', 'w') as f:
        json.dump({'results': results, 'samples': len(df)}, f, indent=2)
    
    md = "# Exp12: Survival Analysis (Quick)\n\n| Model | AUC |\n|-------|-----|\n"
    for r in results:
        md += f"| {r['model']} | {r['auc']:.4f} |\n"
    
    with open(out_dir / 'exp12_survival_quick.md', 'w') as f:
        f.write(md)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
