#!/usr/bin/env python3
"""
Round 4: MLoRA rank 调优 (rank=12/16/32)
在 rank=8 已验证有效的基础上，尝试更大的 rank 看是否有进一步提升
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = f'{PROJECT_DIR}/data_v2'
RESULTS_DIR = f'{PROJECT_DIR}/results'
sys.path.insert(0, PROJECT_DIR)

from models import EmbeddingLayer, TowerNetwork, ProjectionHead


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_domains: int, rank: int = 4):
        super().__init__()
        self.shared = nn.Linear(in_features, out_features)
        self.lora_A = nn.Embedding(n_domains, in_features * rank)
        self.lora_B = nn.Embedding(n_domains, rank * out_features)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        out = self.shared(x)
        A = self.lora_A(domain_ids).view(-1, self.in_features, self.rank)
        B = self.lora_B(domain_ids).view(-1, self.rank, self.out_features)
        lora_out = torch.bmm(torch.bmm(x.unsqueeze(1), A), B).squeeze(1)
        return out + lora_out


class MLoRATower(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], n_domains: int, rank: int = 4):
        super().__init__()
        self.lora_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.lora_layers.append(LoRALinear(dims[i], dims[i+1], n_domains, rank))
            self.activations.append(nn.Sequential(nn.ReLU(), nn.BatchNorm1d(dims[i+1])))
        self.output = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        for lora, act in zip(self.lora_layers, self.activations):
            x = act(lora(x, domain_ids))
        return torch.sigmoid(self.output(x)).squeeze(-1)


class MLoRACVR(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], n_domains: int = 8, rank: int = 4):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_domains = n_domains
        self.rank = rank
        self.tower = None

    def forward(self, x: Dict[str, torch.Tensor], domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = MLoRATower(input_dim, self.hidden_dims, self.n_domains, self.rank).to(embed.device)
        p_cvr = self.tower(embed, domain_ids)
        return {'purchase': p_cvr}


@dataclass
class Config:
    data_dir: str = DATA_DIR
    embed_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    temperature: float = 0.1
    batch_size: int = 4096
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4


class IVRDatasetV2(Dataset):
    def __init__(self, df, features):
        self.feature_data = {}
        for feat in features:
            if feat in df.columns:
                self.feature_data[feat] = torch.tensor(df[feat].values.astype(np.int64), dtype=torch.long)
        self.purchase = torch.tensor(df['purchase'].values.astype(np.float32), dtype=torch.float32)
        self.business_type_id = torch.tensor(df['business_type_id'].values.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.purchase)

    def __getitem__(self, idx):
        return {
            'features': {feat: self.feature_data[feat][idx] for feat in self.feature_data},
            'purchase': self.purchase[idx],
            'business_type_id': self.business_type_id[idx],
        }


def train_and_eval(model_name, model, train_loader, val_loader, bt_ids_val, labels_val, bt_encoder, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model.train()
    for batch in tqdm(train_loader, desc=f'{model_name}'):
        features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
        purchase = batch['purchase'].to(config.device)
        bt_id = batch['business_type_id'].to(config.device)

        outputs = model(features_batch, bt_id)
        loss = F.binary_cross_entropy(outputs['purchase'], purchase)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Eval'):
            features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
            bt_id = batch['business_type_id'].to(config.device)
            outputs = model(features_batch, bt_id)
            all_preds.extend(outputs['purchase'].cpu().numpy().tolist())

    preds = np.array(all_preds)
    labels = labels_val
    bt_ids = bt_ids_val

    overall_auc = roc_auc_score(labels, preds)
    print(f"  Overall AUC: {overall_auc:.4f}")

    bt_aucs = {}
    for bt_id_val in np.unique(bt_ids):
        mask = bt_ids == bt_id_val
        bt_name = bt_encoder.inverse_transform([bt_id_val])[0]
        if bt_name == '__UNKNOWN__':
            continue
        n_pos = labels[mask].sum()
        n_total = mask.sum()
        if n_pos < 5 or n_total - n_pos < 5:
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
    n_domains = len(bt_encoder.classes_)
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, n_domains: {n_domains}")

    config = Config()

    train_dataset = IVRDatasetV2(train_df, features)
    val_dataset = IVRDatasetV2(val_df, features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    labels_val = val_df['purchase'].values.astype(np.float32)
    bt_ids_val = val_df['business_type_id'].values.astype(np.int64)

    ranks = [12, 16, 32]
    all_results = []

    for rank in ranks:
        name = f'mlora_rank{rank}'
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")
        model = MLoRACVR(vocab_sizes, config.embed_dim, config.hidden_dims, n_domains, rank=rank).to(config.device)
        overall_auc, bt_aucs = train_and_eval(name, model, train_loader, val_loader, bt_ids_val, labels_val, bt_encoder, config)
        all_results.append({
            'model': name,
            'rank': rank,
            'overall_auc': round(overall_auc, 4),
            'bt_aucs': bt_aucs
        })

    # Summary
    print("\n" + "="*80)
    print("Round 4 MLoRA Rank Sweep 结果汇总")
    print("="*80)
    bt_names = ['shein', 'aecps', 'aedsp', 'aerta', 'shopee_cps', 'lazada_cps', 'lazada_rta']
    header = f"{'Model':<20} {'Rank':<6} {'Overall':<10}" + "".join(f"{b:<12}" for b in bt_names)
    print(header)
    print("-" * len(header))

    # References
    refs = {
        'baseline': {'overall': 0.8321, 'shein': 0.5150, 'aecps': 0.7387, 'aedsp': 0.6419, 'aerta': 0.7083, 'shopee_cps': 0.7340, 'lazada_cps': 0.7786, 'lazada_rta': 0.7474},
        'mlora_rank8': {'overall': 0.8346, 'shein': 0.5297, 'aecps': 0.7499, 'aedsp': 0.6861, 'aerta': 0.7090, 'shopee_cps': 0.7343, 'lazada_cps': 0.7781, 'lazada_rta': 0.7468},
        'mlora_rank4': {'overall': 0.8329, 'shein': 0.5651, 'aecps': 0.7386, 'aedsp': 0.6769, 'aerta': 0.6953, 'shopee_cps': 0.7357, 'lazada_cps': 0.7757, 'lazada_rta': 0.7522},
    }
    for ref_name, ref_data in refs.items():
        row = f"{ref_name:<20} {'--':<6} {ref_data['overall']:<10}"
        for b in bt_names:
            row += f"{ref_data.get(b, 'N/A'):<12}"
        print(row)

    print("-" * len(header))
    for r in all_results:
        row = f"{r['model']:<20} {r['rank']:<6} {r['overall_auc']:<10}"
        for b in bt_names:
            v = r['bt_aucs'].get(b)
            row += f"{(str(v) if v else 'N/A'):<12}"
        print(row)

    import json
    out_path = f'{RESULTS_DIR}/round4_rank_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")