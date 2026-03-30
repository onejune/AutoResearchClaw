#!/usr/bin/env python3
"""
Round 3: MLoRA 风格 bt-specific 低秩适配层
每个 business_type 有独立的低秩矩阵 (rank=4/8)，叠加在共享 tower 上
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
    """低秩适配线性层：W_out = W_shared + A_bt @ B_bt"""
    def __init__(self, in_features: int, out_features: int, n_domains: int, rank: int = 4):
        super().__init__()
        self.shared = nn.Linear(in_features, out_features)
        # 每个 domain 一对低秩矩阵
        self.lora_A = nn.Embedding(n_domains, in_features * rank)   # [n_domains, in*rank]
        self.lora_B = nn.Embedding(n_domains, rank * out_features)  # [n_domains, rank*out]
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        # Shared path
        out = self.shared(x)
        # LoRA path
        A = self.lora_A(domain_ids).view(-1, self.in_features, self.rank)   # [B, in, rank]
        B = self.lora_B(domain_ids).view(-1, self.rank, self.out_features)  # [B, rank, out]
        lora_out = torch.bmm(torch.bmm(x.unsqueeze(1), A), B).squeeze(1)   # [B, out]
        return out + lora_out


class MLoRATower(nn.Module):
    """带 LoRA 适配的塔网络"""
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
    """MLoRA 风格多 bt 低秩适配 CVR 模型"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], n_domains: int = 8, rank: int = 4):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_domains = n_domains
        self.rank = rank
        self.tower = None  # 延迟初始化

    def forward(self, x: Dict[str, torch.Tensor], domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = MLoRATower(input_dim, self.hidden_dims, self.n_domains, self.rank).to(embed.device)
        p_cvr = self.tower(embed, domain_ids)
        return {'purchase': p_cvr}


class MLoRAUserContrastiveCVR(nn.Module):
    """MLoRA + user_contrastive 组合模型"""
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 32,
                 hidden_dims: List[int] = [256, 128], proj_dim: int = 64,
                 n_domains: int = 8, rank: int = 4, temperature: float = 0.1):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.proj_dim = proj_dim
        self.n_domains = n_domains
        self.rank = rank
        self.temperature = temperature
        self.tower = None
        self.projection = None

    def forward(self, x: Dict[str, torch.Tensor], domain_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        embed = self.embedding(x)
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = MLoRATower(input_dim, self.hidden_dims, self.n_domains, self.rank).to(embed.device)
            self.projection = ProjectionHead(input_dim, self.hidden_dims[0], self.proj_dim).to(embed.device)
        p_cvr = self.tower(embed, domain_ids)
        z = self.projection(embed)
        return {'purchase': p_cvr, 'z': z}

    def contrastive_loss(self, z: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        device = z.device
        batch_size = z.shape[0]
        sim = torch.matmul(z, z.T) / self.temperature
        uids = user_ids.view(-1, 1)
        mask = torch.eq(uids, uids.T).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mask_sum = mask.sum(1)
        valid_mask = mask_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        mean_log_prob = (mask * log_prob).sum(1) / (mask_sum + 1e-8)
        return -mean_log_prob[valid_mask].mean()


@dataclass
class Config:
    data_dir: str = DATA_DIR
    embed_dim: int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    proj_dim: int = 64
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
        if 'user_id' in df.columns:
            self.user_id = torch.tensor(df['user_id'].values.astype(np.int64), dtype=torch.long)
        else:
            self.user_id = None

    def __len__(self):
        return len(self.purchase)

    def __getitem__(self, idx):
        item = {
            'features': {feat: self.feature_data[feat][idx] for feat in self.feature_data},
            'purchase': self.purchase[idx],
            'business_type_id': self.business_type_id[idx],
        }
        if self.user_id is not None:
            item['user_id'] = self.user_id[idx]
        return item


def run_experiment(model_name, model, cl_weight, train_loader, val_loader, bt_ids_val, labels_val, bt_encoder, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model.train()
    for batch in tqdm(train_loader, desc=f'{model_name}'):
        features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
        purchase = batch['purchase'].to(config.device)
        bt_id = batch['business_type_id'].to(config.device)
        user_id = batch.get('user_id')
        if user_id is not None:
            user_id = user_id.to(config.device)

        if isinstance(model, (MLoRACVR, MLoRAUserContrastiveCVR)):
            outputs = model(features_batch, bt_id)
        else:
            outputs = model(features_batch)

        bce_loss = F.binary_cross_entropy(outputs['purchase'], purchase)
        cl_loss = torch.tensor(0.0, device=config.device)
        if 'z' in outputs and user_id is not None and cl_weight > 0:
            cl_loss = model.contrastive_loss(outputs['z'], user_id)

        total_loss = bce_loss + cl_weight * cl_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Eval'):
            features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
            bt_id = batch['business_type_id'].to(config.device)
            if isinstance(model, (MLoRACVR, MLoRAUserContrastiveCVR)):
                outputs = model(features_batch, bt_id)
            else:
                outputs = model(features_batch)
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

    # 预提取 val 标签和 bt_ids
    labels_val = val_df['purchase'].values.astype(np.float32)
    bt_ids_val = val_df['business_type_id'].values.astype(np.int64)

    experiments = [
        # (名称, 模型, cl_weight)
        ('mlora_rank4', MLoRACVR(vocab_sizes, config.embed_dim, config.hidden_dims, n_domains, rank=4).to(config.device), 0.0),
        ('mlora_rank8', MLoRACVR(vocab_sizes, config.embed_dim, config.hidden_dims, n_domains, rank=8).to(config.device), 0.0),
        ('mlora_rank4_user_cl0.1', MLoRAUserContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims, 64, n_domains, rank=4).to(config.device), 0.1),
        ('mlora_rank8_user_cl0.1', MLoRAUserContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims, 64, n_domains, rank=8).to(config.device), 0.1),
    ]

    all_results = []
    for name, model, cl_weight in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")
        overall_auc, bt_aucs = run_experiment(
            name, model, cl_weight, train_loader, val_loader,
            bt_ids_val, labels_val, bt_encoder, config
        )
        all_results.append({
            'model': name,
            'cl_weight': cl_weight,
            'overall_auc': round(overall_auc, 4),
            'bt_aucs': bt_aucs
        })

    # Print summary
    print("\n" + "="*80)
    print("MLoRA 实验结果汇总")
    print("="*80)
    bt_names = ['shein', 'aecps', 'aedsp', 'aerta', 'shopee_cps', 'lazada_cps', 'lazada_rta']
    header = f"{'Model':<35} {'Overall':<10}" + "".join(f"{b:<14}" for b in bt_names)
    print(header)
    print("-" * len(header))
    # Baseline reference
    baseline_ref = {'overall': 0.8321, 'shein': 0.5150, 'aecps': 0.7387, 'aedsp': 0.6419,
                    'aerta': 0.7083, 'shopee_cps': 0.7340, 'lazada_cps': 0.7786, 'lazada_rta': 0.7474}
    row = f"{'baseline(ref)':<35} {baseline_ref['overall']:<10}"
    for b in bt_names:
        row += f"{baseline_ref.get(b, 'N/A'):<14}"
    print(row)
    user_cl_ref = {'overall': 0.8379, 'shein': 0.6078, 'aecps': 0.7430, 'aedsp': 0.6446,
                   'aerta': 0.6837, 'shopee_cps': 0.7379, 'lazada_cps': 0.7765, 'lazada_rta': 0.7495}
    row = f"{'user_contrastive_0.1(ref)':<35} {user_cl_ref['overall']:<10}"
    for b in bt_names:
        row += f"{user_cl_ref.get(b, 'N/A'):<14}"
    print(row)
    print("-" * len(header))
    for r in all_results:
        row = f"{r['model']:<35} {r['overall_auc']:<10}"
        for b in bt_names:
            v = r['bt_aucs'].get(b)
            row += f"{(str(v) if v else 'N/A'):<14}"
        print(row)

    import json
    out_path = f'{RESULTS_DIR}/round3_mlora_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
