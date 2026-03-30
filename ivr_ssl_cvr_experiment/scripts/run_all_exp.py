#!/usr/bin/env python3
"""
data_raw 全部实验：baseline + MLoRA + 对比学习
6 个配置，跑完后输出对比表格
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_DIR = '/mnt/workspace/open_research/autoresearch/ivr_ssl_cvr'
DATA_DIR = f'{PROJECT_DIR}/data_raw'
RESULTS_DIR = f'{PROJECT_DIR}/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ====================== 模型定义 ======================

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_sizes, embed_dim):
        super().__init__()
        self.embs = nn.ModuleDict()
        for k, v in vocab_sizes.items():
            self.embs[k] = nn.Embedding(v, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x_dict):
        embs = []
        for k, emb in self.embs.items():
            embs.append(emb(x_dict[k].clamp(0, emb.num_embeddings - 1)))
        return torch.stack(embs, dim=1)


class BaselineCVR(nn.Module):
    def __init__(self, str_features, num_features, str_vocab_sizes,
                 embed_dim=32, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.str_features = str_features
        self.num_features = num_features
        self.embedding = EmbeddingLayer(str_vocab_sizes, embed_dim)
        self.num_bn = nn.BatchNorm1d(len(num_features))
        total_dim = embed_dim * len(str_features) + len(num_features)
        layers = []
        dims = [total_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_dict):
        str_emb = self.embedding(x_dict)
        str_flat = str_emb.view(str_emb.size(0), -1)
        num_vals = torch.stack([x_dict[f] for f in self.num_features], dim=-1)
        num_norm = self.num_bn(num_vals)
        x = torch.cat([str_flat, num_norm], dim=-1)
        return torch.sigmoid(self.head(self.net(x))).squeeze(-1)


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, n_domains, rank=8):
        super().__init__()
        self.shared = nn.Linear(in_f, out_f, bias=False)
        self.lora_A = nn.Embedding(n_domains, in_f * rank)
        self.lora_B = nn.Embedding(n_domains, rank * out_f)
        self.rank = rank
        self.in_f = in_f

    def forward(self, x, domain_id):
        base = self.shared(x)
        a = self.lora_A(domain_id)  # [B, in_f * rank]
        b = self.lora_B(domain_id)  # [B, rank * out_f]
        r = self.rank
        a = a.view(-1, r, self.in_f).transpose(1, 2)  # [B, in_f, rank]
        b = b.view(-1, r, self.out_f)  # [B, rank, out_f]
        lora = (a @ b).sum(dim=1)  # [B, in_f, out_f]
        return base + lora

    @property
    def out_f(self):
        return self.shared.out_features


class MLoRATower(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_domains, rank=8):
        super().__init__()
        self.lora1 = LoRALinear(embed_dim, hidden_dim, n_domains, rank)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lora2 = LoRALinear(hidden_dim, hidden_dim, n_domains, rank)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, domain_ids):
        x = F.relu(self.bn1(self.lora1(x, domain_ids)))
        return F.relu(self.bn2(self.lora2(x, domain_ids)))


class MLoRACVR(nn.Module):
    def __init__(self, str_features, num_features, str_vocab_sizes, embed_dim=32,
                 tower_dim=128, n_domains=19, rank=8, hidden_dims=[256, 128]):
        super().__init__()
        self.str_features = str_features
        self.num_features = num_features
        self.embedding = EmbeddingLayer(str_vocab_sizes, embed_dim)
        self.num_bn = nn.BatchNorm1d(len(num_features))
        self.n_domains = n_domains
        self.mtowers = nn.ModuleList([MLoRATower(embed_dim, tower_dim, n_domains, rank) for _ in range(len(str_features))])
        self.user_proj = nn.Sequential(nn.Linear(tower_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.head = nn.Linear(tower_dim, 1)

    def _get_user_emb(self, idxs, feat_embs):
        reps = []
        for feat_i, idx in enumerate(idxs):
            reps.append(feat_embs[feat_i, idx])
        return torch.stack(reps, dim=0)

    def forward(self, x_dict, domain_ids, user_ids=None, temperature=0.1):
        # x_dict: {feat: LongTensor [B]}
        feat_embs = self.embedding(x_dict)  # [B, n_feat, embed_dim]
        # MLoRA tower
        tower_outs = []
        for i, tower in enumerate(self.mtowers):
            tower_outs.append(tower(feat_embs[:, i], domain_ids))  # [B, tower_dim]
        x = torch.stack(tower_outs, dim=1).sum(dim=1)  # [B, tower_dim]
        pred = torch.sigmoid(self.head(x)).squeeze(-1)

        result = {'pred': pred}
        if user_ids is not None and self.training:
            # User contrastive
            user_embs = self.user_proj(x)  # [B, embed_dim]
            user_embs = F.normalize(user_embs, dim=-1)
            sim = user_embs @ user_embs.T
            sim = sim / temperature
            labels = (user_ids.unsqueeze(0) == user_ids.unsqueeze(1)).float()
            mask = torch.eye(len(user_ids), device=user_ids.device)
            cl_loss = F.binary_cross_entropy_with_logits(sim[mask == 0], labels[mask == 0])
            result['cl_loss'] = cl_loss
        return result


class RawDataset(Dataset):
    def __init__(self, df, str_features, num_features):
        self.str_data = {f: torch.LongTensor(df[f].values.astype(np.int64)) for f in str_features}
        self.num_data = {f: torch.FloatTensor(df[f].fillna(0).values) for f in num_features}
        self.label = torch.FloatTensor(df['label'].values.astype(np.float32))
        self.bt_id = torch.LongTensor(df['business_type_id'].values.astype(np.int64))
        self.user_id = torch.LongTensor(df['user_id'].values.astype(np.int64))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        features = {}
        for f in self.str_data: features[f] = self.str_data[f][idx]
        for f in self.num_data: features[f] = self.num_data[f][idx]
        return features, self.label[idx], self.bt_id[idx], self.user_id[idx]


def collate_fn(batch):
    features = {k: torch.stack([item[0][k] for item in batch]) for k in batch[0][0].keys()}
    labels = torch.stack([item[1] for item in batch])
    bt_ids = torch.stack([item[2] for item in batch])
    user_ids = torch.stack([item[3] for item in batch])
    return features, labels, bt_ids, user_ids


def evaluate(model, dl, device, use_mlora=False):
    model.eval()
    preds, labels_list, bt_ids_list = [], [], []
    with torch.no_grad():
        for features, labels, bt_ids, user_ids in dl:
            features = {k: v.to(device) for k, v in features.items()}
            if use_mlora:
                p = model(features, bt_ids.to(device))['pred'].cpu().numpy()
            else:
                p = model(features).cpu().numpy()
            preds.extend(p.tolist())
            labels_list.extend(labels.numpy().tolist())
            bt_ids_list.extend(bt_ids.numpy().tolist())
    preds, labels, bt_ids = np.array(preds), np.array(labels_list), np.array(bt_ids_list)
    return roc_auc_score(labels, preds), preds, labels, bt_ids


def per_bt_eval(preds, labels, bt_ids, bt_encoder):
    res = {}
    for bt_id in np.unique(bt_ids):
        mask = bt_ids == bt_id
        if labels[mask].sum() < 5:
            continue
        name = bt_encoder.inverse_transform([bt_id])[0]
        res[name] = {
            'auc': round(roc_auc_score(labels[mask], preds[mask]), 4),
            'n': int(mask.sum()),
            'n_pos': int(labels[mask].sum()),
            'pos_rate': round(labels[mask].sum() / mask.sum(), 4)
        }
    return res


def train_baseline(model, train_dl, device, epochs=1, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for ep in range(epochs):
        for bi, (features, labels, bt_ids, user_ids) in enumerate(train_dl):
            features = {k: v.to(device) for k, v in features.items()}
            pred = model(features)
            loss = F.binary_cross_entropy(pred, labels.to(device))
            opt.zero_grad(); loss.backward(); opt.step()


def train_mlora(model, train_dl, device, cl_weight=0.0, epochs=1, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for ep in range(epochs):
        for bi, (features, labels, bt_ids, user_ids) in enumerate(train_dl):
            features = {k: v.to(device) for k, v in features.items()}
            bt_d = bt_ids.to(device)
            out = model(features, bt_d, user_ids.to(device), temperature=0.1)
            loss = F.binary_cross_entropy(out['pred'], labels.to(device))
            if cl_weight > 0:
                loss = loss + cl_weight * out.get('cl_loss', torch.tensor(0.0).to(device))
            opt.zero_grad(); loss.backward(); opt.step()


def main():
    print("="*60)
    print("data_raw 全实验")
    print("="*60)

    # 加载数据
    print("\n加载数据...")
    train_df = pq.read_table(f'{DATA_DIR}/train.parquet').to_pandas()
    test_df = pq.read_table(f'{DATA_DIR}/test.parquet').to_pandas()
    with open(f'{DATA_DIR}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    str_features = meta['str_features']
    num_features = meta['num_features']
    str_vocab_sizes = meta['vocab_sizes']
    bt_encoder = meta['bt_encoder']
    n_domains = len(bt_encoder.classes_)

    print(f"训练: {len(train_df):,}, 测试: {len(test_df):,}")
    print(f"str_feat={len(str_features)}, num_feat={len(num_features)}, n_domains={n_domains}")

    train_ds = RawDataset(train_df, str_features, num_features)
    test_ds = RawDataset(test_df, str_features, num_features)
    BS = 8192
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=4, collate_fn=collate_fn)

    experiments = [
        ('baseline', lambda: BaselineCVR(str_features, num_features, str_vocab_sizes, 32, [512,256,128])),
        ('mlora_rank8', lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 8, [256,128])),
        ('mlora_rank12', lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 12, [256,128])),
        ('user_contrastive', lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 4, [256,128])),
        ('mlora_rank8_user_cl0.1', lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 8, [256,128])),
        ('mlora_rank12_user_cl0.1', lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 12, [256,128])),
    ]
    cl_weights = {'baseline': 0, 'mlora_rank8': 0, 'mlora_rank12': 0,
                   'user_contrastive': 0.1, 'mlora_rank8_user_cl0.1': 0.1, 'mlora_rank12_user_cl0.1': 0.1}

    all_results = {}
    for name, model_fn in experiments:
        print(f"\n{'='*60}")
        print(f"训练: {name}")
        print(f"{'='*60}")
        model = model_fn().to(DEVICE)
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        use_mlora = 'mlora' in name or 'user_contrastive' in name
        cl_w = cl_weights[name]
        if use_mlora:
            train_mlora(model, train_dl, DEVICE, cl_weight=cl_w, epochs=1)
        else:
            train_baseline(model, train_dl, DEVICE, epochs=1)
        auc, preds, labels, bt_ids = evaluate(model, test_dl, DEVICE, use_mlora=use_mlora)
        bt_res = per_bt_eval(preds, labels, bt_ids, bt_encoder)
        print(f"Overall AUC: {auc:.4f}")
        all_results[name] = {'overall_auc': round(auc, 4), 'per_bt': bt_res, 'config': {'cl_weight': cl_w}}

    # 保存结果
    with open(f'{RESULTS_DIR}/data_raw_all_exp.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成 markdown 对比表格
    bt_order = list(bt_encoder.classes_)
    header = "| BT | " + " | ".join([n for n in all_results]) + " |"
    sep = "|------|" + "|------:" * len(all_results) + "|"
    rows = []
    for bt in bt_order:
        aucs = []
        for n in all_results:
            r = all_results[n]['per_bt'].get(bt, {})
            aucs.append(f"{r.get('auc', '-'):.4f}" if 'auc' in r else '-')
        rows.append(f"| {bt} | " + " | ".join(aucs) + " |")
    overall_row = "| **Overall** | " + " | ".join([f"**{all_results[n]['overall_auc']:.4f}**" for n in all_results]) + " |"

    md = f"""# data_raw 全实验结果

## 配置
- embed_dim=32, hidden=[512,256,128], 1 epoch, batch_size={BS}
- user_contrastive / mlora_rank*_user_cl0.1: cl_weight=0.1

## per-BT AUC 对比

{header}
{sep}
""" + "\n".join(rows) + f"\n{overall_row}\n"
    with open(f'{RESULTS_DIR}/data_raw_all_exp.md', 'w') as f:
        f.write(md)

    # 打印汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    print(header)
    print(sep)
    print(overall_row)

    print(f"\n结果已保存: {RESULTS_DIR}/data_raw_all_exp.json")


if __name__ == '__main__':
    main()