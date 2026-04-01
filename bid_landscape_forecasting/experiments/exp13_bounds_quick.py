"""
Exp13: Boundary-Enhanced Survival Analysis (Quick Test)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

project_root = Path(__file__).parent.parent


class DeepHitBounds(nn.Module):
    def __init__(self, input_dim, K=30):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, K), nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def survival(self, pmf):
        return torch.cumsum(torch.flip(pmf, [-1]), dim=-1).flip([-1])
    
    def loss(self, pmf, times, events, beta=0.2):
        N, K = pmf.shape
        eps = 1e-10
        
        # Likelihood
        ll = 0.0
        boundary_loss = 0.0
        b_count = 0
        
        for i in range(N):
            t = times[i]
            bin_idx = min(int(t * K), K-1)
            
            if events[i] == 1:
                ll += torch.log(pmf[i, bin_idx] + eps)
            else:
                if bin_idx < K-1:
                    surv = pmf[i, bin_idx+1:].sum() + eps
                else:
                    surv = torch.tensor(eps, device=pmf.device)
                ll += torch.log(surv)
                
                # Boundary constraint: S(bid-ε) ≈ 1
                left = max(0, bin_idx - 2)
                if left > 0:
                    surv_left = pmf[i, left:].sum()
                    boundary_loss += (1.0 - surv_left) ** 2
                    b_count += 1
        
        ll_loss = -ll / N
        bound_loss = (boundary_loss / b_count) if b_count > 0 else 0
        
        return ll_loss + beta * bound_loss, ll_loss, bound_loss


def main():
    print("="*50)
    print("Exp13: Boundary-Enhanced Survival (Quick)")
    print("="*50)
    
    # Load data
    df = pd.read_parquet(project_root / 'data' / 'bid_landscape_train_small.parquet')
    if len(df) > 15000:
        df = df.sample(15000, random_state=42)
    
    print(f"Samples: {len(df)}")
    
    # Features
    features = ['bid_amount', 'business_type', 'deviceid', 'adid']
    X = df[features].fillna(0).values.astype(np.float32)
    
    # Survival format
    events = df['win_label'].values.astype(np.float32)
    times = np.where(events == 1, 
                     df['true_value'].values,
                     df['bid_amount'].values).astype(np.float32)
    
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    times = (times - times.min()) / (times.max() - times.min() + 1e-10)
    
    print(f"Win rate: {events.mean():.4f}")
    
    # Split
    X_tr, X_va, t_tr, t_va, e_tr, e_va = \
        train_test_split(X, times, events, test_size=0.2, random_state=42, stratify=events)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    t_tr_t = torch.FloatTensor(t_tr).to(device)
    e_tr_t = torch.FloatTensor(e_tr).to(device)
    X_va_t = torch.FloatTensor(X_va).to(device)
    
    results = []
    
    # === Baseline: DeepHit without bounds ===
    print("\n[1/2] Training DeepHit (baseline)...")
    model = DeepHitBounds(X_tr.shape[1], K=30).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    loader = DataLoader(TensorDataset(X_tr_t, t_tr_t, e_tr_t), batch_size=512, shuffle=True)
    
    best_auc = 0
    for epoch in range(15):
        model.train()
        for bx, bt, be in loader:
            opt.zero_grad()
            pmf = model(bx)
            loss, _, _ = model.loss(pmf, bt, be, beta=0.0)  # No boundary loss
            loss.backward()
            opt.step()
        
        model.eval()
        with torch.no_grad():
            pmf = model(X_va_t)
            surv = model.survival(pmf)
            probs = surv[:, 15].cpu().numpy()
            auc = roc_auc_score(e_va, np.clip(probs, 0, 1))
            if auc > best_auc:
                best_auc = auc
    
    print(f"  Best AUC: {best_auc:.4f}")
    results.append({'model': 'DeepHit (baseline)', 'auc': best_auc})
    
    # === Enhanced: DeepHit with boundary constraints ===
    print("\n[2/2] Training DeepHit + Bounds...")
    model = DeepHitBounds(X_tr.shape[1], K=30).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    
    best_auc = 0
    for epoch in range(15):
        model.train()
        total_ll = 0
        total_bound = 0
        
        for bx, bt, be in loader:
            opt.zero_grad()
            pmf = model(bx)
            loss, ll, bound = model.loss(pmf, bt, be, beta=0.25)
            loss.backward()
            opt.step()
            total_ll += ll.item()
            total_bound += bound.item()
        
        model.eval()
        with torch.no_grad():
            pmf = model(X_va_t)
            surv = model.survival(pmf)
            probs = surv[:, 15].cpu().numpy()
            auc = roc_auc_score(e_va, np.clip(probs, 0, 1))
            if auc > best_auc:
                best_auc = auc
        
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}: LL={total_ll/len(loader):.4f}, Bound={total_bound/len(loader):.4f}, AUC={auc:.4f}")
    
    print(f"  Best AUC: {best_auc:.4f}")
    results.append({'model': 'DeepHit + Bounds', 'auc': best_auc})
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for r in results:
        print(f"{r['model']:25s} | AUC: {r['auc']:.4f}")
    
    # Save
    out_dir = project_root / 'results'
    out_dir.mkdir(exist_ok=True)
    
    import json
    with open(out_dir / 'exp13_bounds_quick.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    md = "# Exp13: Boundary-Enhanced Survival (Quick)\n\n| Model | AUC |\n|-------|-----|\n"
    for r in results:
        md += f"| {r['model']} | {r['auc']:.4f} |\n"
    
    with open(out_dir / 'exp13_bounds_quick.md', 'w') as f:
        f.write(md)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
