#!/usr/bin/env python3
"""
data_v3 剩余实验：实验 4~6
  4. user_contrastive（cl_weight=0.1）
  5. mlora_rank8_user_cl0.1（rank=8 + user_contrastive）
  6. mlora_rank12_user_cl0.1（rank=12 + user_contrastive）
"""
import os, sys, pickle, json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = f'{PROJECT_DIR}/data_v3'
RESULTS_DIR = f'{PROJECT_DIR}/results'
sys.path.insert(0, PROJECT_DIR)

from models import EmbeddingLayer, TowerNetwork, ProjectionHead


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, n_domains, rank=4):
        super().__init__()
        self.shared = nn.Linear(in_features, out_features)
        self.lora_A = nn.Embedding(n_domains, in_features * rank)
        self.lora_B = nn.Embedding(n_domains, rank * out_features)
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, domain_ids):
        out = self.shared(x)
        A = self.lora_A(domain_ids).view(-1, self.in_features, self.rank)
        B = self.lora_B(domain_ids).view(-1, self.rank, self.out_features)
        lora_out = torch.bmm(torch.bmm(x.unsqueeze(1), A), B).squeeze(1)
        return out + lora_out


class MLoRATower(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_domains, rank=4):
        super().__init__()
        self.lora_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.lora_layers.append(LoRALinear(dims[i], dims[i+1], n_domains, rank))
            self.activations.append(nn.Sequential(nn.ReLU(), nn.BatchNorm1d(dims[i+1])))
        self.output = nn.Linear(dims[-1], 1)

    def forward(self, x, domain_ids):
        for lora, act in zip(self.lora_layers, self.activations):
            x = act(lora(x, domain_ids))
        return torch.sigmoid(self.output(x)).squeeze(-1)


class MLoRACVR(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=32, hidden_dims=[256, 128], n_domains=8, rank=4):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.embedding = EmbeddingLayer(vocab_sizes, embed_dim)
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_domains = n_domains
        self.rank = rank
        self.tower = None

    def forward(self, x, domain_ids):
        embed = self.embedding(x)
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = MLoRATower(input_dim, self.hidden_dims, self.n_domains, self.rank).to(embed.device)
        p_cvr = self.tower(embed, domain_ids)
        return {'purchase': p_cvr}


class MLoRAUserContrastiveCVR(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=32, hidden_dims=[256, 128], proj_dim=64,
                 n_domains=8, rank=4, temperature=0.1):
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

    def forward(self, x, domain_ids):
        embed = self.embedding(x)
        if self.tower is None:
            input_dim = embed.shape[-1]
            self.tower = MLoRATower(input_dim, self.hidden_dims, self.n_domains, self.rank).to(embed.device)
            self.projection = ProjectionHead(input_dim, self.hidden_dims[0], self.proj_dim).to(embed.device)
        p_cvr = self.tower(embed, domain_ids)
        z = self.projection(embed)
        return {'purchase': p_cvr, 'z': z}

    def contrastive_loss(self, z, user_ids):
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


class IVRDatasetV3(Dataset):
    def __init__(self, df, features):
        self.feature_data = {}
        for feat in features:
            if feat in df.columns:
                self.feature_data[feat] = torch.tensor(df[feat].values.astype(np.int64), dtype=torch.long)
        self.label = torch.tensor(df['label'].values.astype(np.float32), dtype=torch.float32)
        self.business_type_id = torch.tensor(df['business_type_id'].values.astype(np.int64), dtype=torch.long)
        if 'user_id' in df.columns:
            self.user_id = torch.tensor(df['user_id'].values.astype(np.int64), dtype=torch.long)
        else:
            self.user_id = None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        item = {
            'features': {feat: self.feature_data[feat][idx] for feat in self.feature_data},
            'label': self.label[idx],
            'business_type_id': self.business_type_id[idx],
        }
        if self.user_id is not None:
            item['user_id'] = self.user_id[idx]
        return item


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


def run_experiment(model_name, model, cl_weight, train_loader, test_loader,
                   bt_ids_test, labels_test, bt_encoder, config):
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name}  (cl_weight={cl_weight})")
    print(f"Device: {config.device}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(train_loader, desc=f'Train {model_name}'):
        features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
        label = batch['label'].to(config.device)
        bt_id = batch['business_type_id'].to(config.device)
        user_id = batch.get('user_id')
        if user_id is not None:
            user_id = user_id.to(config.device)

        if isinstance(model, (MLoRACVR, MLoRAUserContrastiveCVR)):
            outputs = model(features_batch, bt_id)
        else:
            outputs = model(features_batch)

        bce_loss = F.binary_cross_entropy(outputs['purchase'], label)
        cl_loss = torch.tensor(0.0, device=config.device)
        if 'z' in outputs and user_id is not None and cl_weight > 0:
            cl_loss = model.contrastive_loss(outputs['z'], user_id)

        loss = bce_loss + cl_weight * cl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Avg train loss: {avg_loss:.4f}")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Eval {model_name}'):
            features_batch = {k: v.to(config.device) for k, v in batch['features'].items()}
            bt_id = batch['business_type_id'].to(config.device)
            if isinstance(model, (MLoRACVR, MLoRAUserContrastiveCVR)):
                outputs = model(features_batch, bt_id)
            else:
                outputs = model(features_batch)
            all_preds.extend(outputs['purchase'].cpu().numpy().tolist())

    preds = np.array(all_preds)
    labels = labels_test
    bt_ids = bt_ids_test

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
    print("=" * 70)
    print("data_v3 剩余实验 (实验 4~6)")
    print("=" * 70)

    print("\nLoading data...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, 'test.parquet'))
    with open(os.path.join(DATA_DIR, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)

    features = meta['features']
    vocab_sizes = meta['vocab_sizes']
    bt_encoder = meta['bt_encoder']
    n_domains = len(bt_encoder.classes_)
    print(f"Train: {len(train_df):,}, Test: {len(test_df):,}")
    print(f"n_domains: {n_domains}, Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    config = Config()

    train_dataset = IVRDatasetV3(train_df, features)
    test_dataset = IVRDatasetV3(test_df, features)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True)

    labels_test = test_df['label'].values.astype(np.float32)
    bt_ids_test = test_df['business_type_id'].values.astype(np.int64)

    from models import UserContrastiveCVR

    experiments = [
        ('user_contrastive',
         lambda: UserContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims, config.proj_dim, config.temperature).to(config.device),
         0.1),
        ('mlora_rank8_user_cl0.1',
         lambda: MLoRAUserContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims, config.proj_dim, n_domains, rank=8).to(config.device),
         0.1),
        ('mlora_rank12_user_cl0.1',
         lambda: MLoRAUserContrastiveCVR(vocab_sizes, config.embed_dim, config.hidden_dims, config.proj_dim, n_domains, rank=12).to(config.device),
         0.1),
    ]

    # Load existing results (experiments 1-3)
    json_path = os.path.join(RESULTS_DIR, 'data_v3_results.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing results")
    else:
        all_results = []

    for name, model_fn, cl_weight in experiments:
        model = model_fn()
        overall_auc, bt_aucs = run_experiment(
            name, model, cl_weight, train_loader, test_loader,
            bt_ids_test, labels_test, bt_encoder, config
        )
        all_results.append({
            'model': name,
            'cl_weight': cl_weight,
            'overall_auc': round(overall_auc, 4),
            'bt_aucs': bt_aucs
        })
        # Save after each experiment
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  Saved intermediate results to {json_path}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Generate Markdown summary
    bt_names = [b for b in bt_encoder.classes_ if b != '__UNKNOWN__']
    lines = []
    lines.append("# data_v3 实验结果汇总")
    lines.append(f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n数据集：data_v3（训练集 {len(train_df):,} 行，测试集 {len(test_df):,} 行）")
    lines.append("\n## Overall AUC 对比\n")
    header = "| 模型 | Overall AUC |" + "".join(f" {b} |" for b in bt_names)
    sep = "|------|-------------|" + "".join("------|" for _ in bt_names)
    lines.append(header)
    lines.append(sep)
    for r in all_results:
        row = f"| {r['model']} | {r['overall_auc']:.4f} |"
        for b in bt_names:
            v = r['bt_aucs'].get(b, 'N/A')
            row += f" {v} |"
        lines.append(row)

    lines.append("\n## Per-BT AUC 详情\n")
    for r in all_results:
        lines.append(f"### {r['model']}")
        lines.append(f"- Overall AUC: **{r['overall_auc']:.4f}**")
        lines.append(f"- cl_weight: {r['cl_weight']}")
        lines.append("")
        lines.append("| business_type | AUC |")
        lines.append("|---------------|-----|")
        for bt_name, auc in sorted(r['bt_aucs'].items()):
            lines.append(f"| {bt_name} | {auc} |")
        lines.append("")

    md_path = os.path.join(RESULTS_DIR, 'data_v3_results_summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved Markdown: {md_path}")

    # Console summary
    print("\n" + "=" * 80)
    print("data_v3 实验结果汇总")
    print("=" * 80)
    col_w = 12
    header_str = f"{'Model':<35} {'Overall':<10}" + "".join(f"{b:<{col_w}}" for b in bt_names)
    print(header_str)
    print("-" * len(header_str))
    for r in all_results:
        row_str = f"{r['model']:<35} {r['overall_auc']:<10.4f}"
        for b in bt_names:
            v = r['bt_aucs'].get(b)
            row_str += f"{(str(v) if v else 'N/A'):<{col_w}}"
        print(row_str)
    print("=" * 80)
    print("\nDone!")
