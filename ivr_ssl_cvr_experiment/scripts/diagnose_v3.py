#!/usr/bin/env python3
"""
data_v3 诊断实验：找出为什么模型泛化差
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import defaultdict

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = f'{PROJECT_DIR}/data_v3'

# ============================================================
# 诊断1: 简单 baseline（减小 vocab，观察是否改善）
# ============================================================
def diag_vocab_reduction():
    print("\n" + "="*60)
    print("诊断1: Vocab 大小对泛化能力的影响")
    print("="*60)
    
    t = pq.read_table(f'{DATA_DIR}/train.parquet')
    df = t.to_pandas()
    feat_cols = [c for c in df.columns if c not in ['label','business_type_id','user_id']]
    
    # 采样
    idx = np.random.RandomState(42).choice(len(df), 100000, replace=False)
    X = df.iloc[idx][feat_cols].values.astype(np.int64) % 1000  # 强制 vocab=1000
    y = df.iloc[idx]['label'].values.astype(np.float32)
    
    print(f'强制 vocab=1000, 测试特征范围: [{X.min()}, {X.max()}]')
    
    class Simple(nn.Module):
        def __init__(self, n_feat=124, vocab=1000, emb_dim=16):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(vocab, emb_dim) for _ in range(n_feat)])
            self.linear = nn.Linear(emb_dim * n_feat, 128)
            self.out = nn.Linear(128, 1)
        def forward(self, x):
            out = torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embs)], dim=1)
            out = out.view(x.size(0), -1)
            return torch.sigmoid(self.out(F.relu(self.linear(out)))).squeeze(-1)
    
    model = Simple(124, 1000, 16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.LongTensor(X), torch.FloatTensor(y))
    dl = DataLoader(ds, batch_size=4096, shuffle=True)
    
    for i, (bx, by) in enumerate(dl):
        if i >= 100: break
        pred = model(bx)
        loss = F.binary_cross_entropy(pred, by)
        opt.zero_grad(); loss.backward(); opt.step()
    
    model.eval()
    preds = []
    with torch.no_grad():
        for bx, by in dl:
            preds.append(model(bx).numpy())
    
    preds = np.concatenate(preds)
    auc = roc_auc_score(y, preds)
    print(f'强制 vocab=1000, 训练100batch后 AUC: {auc:.4f}')
    print(f'预测范围: [{preds.min():.4f}, {preds.max():.4f}]')


# ============================================================
# 诊断2: 特征重要性（看每个特征单独能预测多少）
# ============================================================
def diag_single_feature_auc():
    print("\n" + "="*60)
    print("诊断2: 单特征 AUC（找出哪些特征有预测能力）")
    print("="*60)
    
    t = pq.read_table(f'{DATA_DIR}/test.parquet')
    df = t.to_pandas()
    feat_cols = [c for c in df.columns if c not in ['label','business_type_id','user_id']]
    y = df['label'].values
    
    # 采样 5 万
    idx = np.random.RandomState(42).choice(len(df), 50000, replace=False)
    df_sampled = df.iloc[idx]
    y_s = y[idx]
    
    results = []
    for col in tqdm(feat_cols, desc='Computing single-feature AUC'):
        x = df_sampled[col].values.astype(np.float32)
        try:
            auc = roc_auc_score(y_s, x)
            results.append((col, abs(auc - 0.5), auc))
        except:
            results.append((col, 0, 0.5))
    
    results.sort(key=lambda x: -x[1])
    print("\n单特征 AUC（Top 20）:")
    for col, score, auc in results[:20]:
        direction = "↑" if auc > 0.5 else "↓"
        print(f'  {col:50s} AUC={auc:.4f} {direction} (|Δ|={score:.4f})')
    
    print(f"\n特征 AUC 分布:")
    aucs = [r[2] for r in results]
    print(f'  >0.52 (有信号): {sum(1 for a in aucs if abs(a-0.5)>0.02)}')
    print(f'  0.50~0.52 (弱): {sum(1 for a in aucs if 0.02>=abs(a-0.5)>0.005)}')
    print(f'  ≈0.50 (无信号): {sum(1 for a in aucs if abs(a-0.5)<=0.005)}')


# ============================================================
# 诊断3: 实际运行 baseline 模型 1 epoch，观察过拟合程度
# ============================================================
def diag_full_baseline():
    print("\n" + "="*60)
    print("诊断3: 完整 baseline 模型 (1 epoch)")
    print("="*60)
    
    with open(f'{DATA_DIR}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    
    # 训练集采样 50 万
    train_t = pq.read_table(f'{DATA_DIR}/train.parquet')
    train_df = train_t.to_pandas()
    idx = np.random.RandomState(42).choice(len(train_df), 500000, replace=False)
    train_df_s = train_df.iloc[idx]
    
    # 测试集采样 20 万
    test_t = pq.read_table(f'{DATA_DIR}/test.parquet')
    test_df = test_t.to_pandas()
    idx2 = np.random.RandomState(42).choice(len(test_df), 200000, replace=False)
    test_df_s = test_df.iloc[idx2]
    
    feat_cols = meta['features']
    vocab_sizes = {k: 1000 for k in feat_cols}  # 限制 vocab=1000 防止过拟合
    
    class Baseline(nn.Module):
        def __init__(self, vocab_sizes, embed_dim=16, hidden_dims=[128,64]):
            super().__init__()
            self.embs = nn.ModuleDict()
            for k, v in vocab_sizes.items():
                self.embs[k] = nn.Embedding(min(v, 1000), embed_dim)
            total_dim = embed_dim * len(vocab_sizes)
            layers = []
            dims = [total_dim] + hidden_dims
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dims[-1], 1)
        
        def forward(self, x_dict):
            embs = []
            for k in self.embs:
                embs.append(self.embs[k](x_dict[k].clamp(0, 999)))
            x = torch.cat(embs, dim=-1)
            return torch.sigmoid(self.head(self.net(x))).squeeze(-1)
    
    model = Baseline(vocab_sizes, embed_dim=16, hidden_dims=[128,64])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_x = {k: torch.LongTensor(train_df_s[k].values % 1000) for k in feat_cols}
    train_y = torch.FloatTensor(train_df_s['label'].values)
    test_x = {k: torch.LongTensor(test_df_s[k].values % 1000) for k in feat_cols}
    test_y = torch.FloatTensor(test_df_s['label'].values)
    
    BS = 8192
    train_ds = TensorDataset(*[train_x[k] for k in feat_cols], train_y)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    
    test_ds = TensorDataset(*[test_x[k] for k in feat_cols], test_y)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False)
    
    print(f"训练 {len(train_df_s):,} 样本, 测试 {len(test_df_s):,} 样本")
    
    # 1 epoch 训练
    model.train()
    for bi, (bx, *rest) in enumerate(train_dl):
        by = rest[-1]
        x_dict = {feat_cols[i]: bx[:, i] for i in range(len(feat_cols))}
        pred = model(x_dict)
        loss = F.binary_cross_entropy(pred, by)
        opt.zero_grad(); loss.backward(); opt.step()
        if bi % 200 == 0:
            print(f"  batch {bi}/{len(train_dl)}: loss={loss.item():.4f}")
    
    # 评估
    model.eval()
    train_preds, test_preds = [], []
    with torch.no_grad():
        for bx, *rest in train_dl:
            by = rest[-1]
            x_dict = {feat_cols[i]: bx[:, i] for i in range(len(feat_cols))}
            train_preds.append(model(x_dict).numpy())
        for bx, *rest in test_dl:
            x_dict = {feat_cols[i]: bx[:, i] for i in range(len(feat_cols))}
            test_preds.append(model(x_dict).numpy())
    
    train_auc = roc_auc_score(train_y.numpy(), np.concatenate(train_preds))
    test_auc = roc_auc_score(test_y.numpy(), np.concatenate(test_preds))
    print(f"\n训练集 AUC: {train_auc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"Gap: {train_auc - test_auc:.4f} ({'严重过拟合' if train_auc - test_auc > 0.1 else '正常'})")
    
    # per-BT AUC
    bt_encoder = meta['bt_encoder']
    for bt_id, bt_name in enumerate(bt_encoder.classes_):
        if bt_name == '__UNKNOWN__':
            continue
        mask = test_df_s['business_type_id'] == bt_id
        if mask.sum() < 50:
            continue
        auc = roc_auc_score(test_y[mask].numpy(), np.array(np.concatenate(test_preds))[mask])
        print(f"  [{bt_name}] AUC={auc:.4f}  (n={mask.sum():,})")


if __name__ == '__main__':
    diag_vocab_reduction()
    diag_single_feature_auc()
    diag_full_baseline()