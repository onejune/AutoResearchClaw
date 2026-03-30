#!/usr/bin/env python3
"""
按 business_type 分组评估各实验的 AUC
直接重跑推理（不重新训练），复用 run_all_v2_fixed.py 的模型结构
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = f'{PROJECT_DIR}/data_v2'
RESULTS_DIR = f'{PROJECT_DIR}/results'
sys.path.insert(0, PROJECT_DIR)

from models import build_model

@dataclass
class Config:
    data_dir: str = DATA_DIR
    model_name: str = 'baseline'
    embed_dim: int = 32
    hidden_dims: List[int] = None
    proj_dim: int = 64
    temperature: float = 0.1
    batch_size: int = 4096
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    contrastive_weight: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class IVRDatasetV2(Dataset):
    def __init__(self, df, features):
        self.features = features
        self.data = {}
        for feat in features:
            if feat in df.columns:
                self.data[feat] = torch.tensor(df[feat].values.astype(np.int64), dtype=torch.long)
        self.purchase = torch.tensor(df['purchase'].values.astype(np.float32), dtype=torch.float32)
        self.business_type_id = torch.tensor(df['business_type_id'].values.astype(np.int64), dtype=torch.long)
        if 'user_id' in df.columns:
            self.user_id = torch.tensor(df['user_id'].values.astype(np.int64), dtype=torch.long)
        else:
            self.user_id = None

    def __len__(self):
        return len(self.purchase)

    def __getitem__(self, idx):
        features = {feat: self.data[feat][idx] for feat in self.data}
        item = {
            'features': features,
            'purchase': self.purchase[idx],
            'business_type_id': self.business_type_id[idx],
        }
        if self.user_id is not None:
            item['user_id'] = self.user_id[idx]
        return item


def train_and_eval_by_bt(model_name, cl_weight, train_df, val_df, features, vocab_sizes, bt_encoder):
    config = Config()
    config.model_name = model_name
    config.contrastive_weight = cl_weight

    train_dataset = IVRDatasetV2(train_df, features)
    val_dataset = IVRDatasetV2(val_df, features)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(model_name, vocab_sizes, config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Train
    model.train()
    for batch in tqdm(train_loader, desc=f'Training {model_name}(cw={cl_weight})'):
        features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
        purchase = batch['purchase'].to(config.device)
        bt_id = batch['business_type_id'].to(config.device)
        user_id = batch.get('user_id')
        if user_id is not None:
            user_id = user_id.to(config.device)

        outputs = model(features_batch)
        bce_loss = F.binary_cross_entropy(outputs['purchase'], purchase)

        cl_loss = torch.tensor(0.0, device=config.device)
        if model_name == 'augment_contrastive' and 'z1' in outputs:
            cl_loss = model.contrastive_loss(outputs['z1'], outputs['z2'])
        elif model_name == 'user_contrastive' and 'z' in outputs and user_id is not None:
            cl_loss = model.contrastive_loss(outputs['z'], user_id)
        elif model_name == 'bt_contrastive' and 'z' in outputs:
            cl_loss = model.contrastive_loss(outputs['z'], bt_id)
        elif model_name == 'contrastive' and 'z' in outputs:
            cl_loss = model.contrastive_loss(outputs['z'], purchase.long())

        total_loss = bce_loss + cl_weight * cl_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # Eval with business_type tracking
    model.eval()
    all_preds, all_labels, all_bt_ids = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Evaluating {model_name}'):
            features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
            outputs = model(features_batch)
            all_preds.extend(outputs['purchase'].cpu().numpy().tolist())
            all_labels.extend(batch['purchase'].numpy().tolist())
            all_bt_ids.extend(batch['business_type_id'].numpy().tolist())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    bt_ids = np.array(all_bt_ids)

    # Overall AUC
    overall_auc = roc_auc_score(labels, preds)
    print(f"  Overall AUC: {overall_auc:.4f}")

    # Per business_type AUC
    bt_aucs = {}
    for bt_id in np.unique(bt_ids):
        mask = bt_ids == bt_id
        bt_name = bt_encoder.inverse_transform([bt_id])[0]
        n_pos = labels[mask].sum()
        n_total = mask.sum()
        if n_pos < 5 or n_total - n_pos < 5:
            print(f"  [{bt_name}] skip (pos={int(n_pos)}, total={int(n_total)})")
            bt_aucs[bt_name] = None
            continue
        auc = roc_auc_score(labels[mask], preds[mask])
        bt_aucs[bt_name] = round(auc, 4)
        print(f"  [{bt_name}] AUC={auc:.4f}  (n={int(n_total)}, pos_rate={n_pos/n_total:.3%})")

    return overall_auc, bt_aucs


if __name__ == '__main__':
    print("Loading data...")
    train_df = pd.read_pickle(os.path.join(DATA_DIR, 'train.pkl'))
    val_df = pd.read_pickle(os.path.join(DATA_DIR, 'val.pkl'))
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    features = meta['features']
    vocab_sizes = meta['vocab_sizes']
    bt_encoder = meta['bt_encoder']
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")
    print(f"Business types: {bt_encoder.classes_}")

    experiments = [
        ('baseline', 0.1),
        ('bt_contrastive', 0.05),
        ('bt_contrastive', 0.1),
        ('bt_contrastive', 0.2),
        ('user_contrastive', 0.1),
        ('augment_contrastive', 0.1),
    ]

    all_results = []
    for model_name, cl_weight in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {model_name}  cl_weight={cl_weight}")
        print(f"{'='*60}")
        overall_auc, bt_aucs = train_and_eval_by_bt(
            model_name, cl_weight, train_df, val_df, features, vocab_sizes, bt_encoder
        )
        all_results.append({
            'model': model_name,
            'cl_weight': cl_weight,
            'overall_auc': round(overall_auc, 4),
            'bt_aucs': bt_aucs
        })

    # Print summary table
    print("\n" + "="*80)
    print("按 business_type 的 AUC 汇总")
    print("="*80)

    bt_names = [b for b in bt_encoder.classes_ if b != '__UNKNOWN__']
    header = f"{'Model':<30} {'Overall':<10}" + "".join(f"{b:<14}" for b in bt_names)
    print(header)
    print("-" * len(header))
    for r in all_results:
        label = f"{r['model']}(cw={r['cl_weight']})"
        row = f"{label:<30} {r['overall_auc']:<10}"
        for b in bt_names:
            v = r['bt_aucs'].get(b)
            row += f"{(str(v) if v else 'N/A'):<14}"
        print(row)

    # Save
    out_path = f'{RESULTS_DIR}/eval_by_bt_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
