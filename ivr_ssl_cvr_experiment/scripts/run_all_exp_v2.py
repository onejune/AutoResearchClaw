#!/usr/bin/env python3
"""
data_raw 全部实验：baseline + MLoRA + 对比学习（共10个配置）
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
    """普通 baseline：embedding + 数值特征 concat + MLP"""
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

    def get_repr(self, x_dict):
        """返回 128 维表示，用于对比学习"""
        str_emb = self.embedding(x_dict)
        str_flat = str_emb.view(str_emb.size(0), -1)
        num_vals = torch.stack([x_dict[f] for f in self.num_features], dim=-1)
        num_norm = self.num_bn(num_vals)
        x = torch.cat([str_flat, num_norm], dim=-1)
        return self.net[:-1](x)  # [B, 128]


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
        a = self.lora_A(domain_id)
        b = self.lora_B(domain_id)
        r = self.rank
        a = a.view(-1, r, self.in_f).transpose(1, 2)
        b = b.view(-1, r, self.out_f)
        lora = (a @ b).sum(dim=1)
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
    """MLoRA: 每个字符串特征过一个 bt-specific LoRA tower"""
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

    def forward(self, x_dict, domain_ids, user_ids=None, temperature=0.1, cl_type=None, cl_weight=0.0):
        feat_embs = self.embedding(x_dict)
        tower_outs = []
        for i, tower in enumerate(self.mtowers):
            tower_outs.append(tower(feat_embs[:, i], domain_ids))
        x = torch.stack(tower_outs, dim=1).sum(dim=1)
        pred = torch.sigmoid(self.head(x)).squeeze(-1)

        result = {'pred': pred}
        if self.training and cl_weight > 0:
            user_embs = self.user_proj(x)
            user_embs = F.normalize(user_embs, dim=-1)
            sim = user_embs @ user_embs.T
            sim = sim / temperature
            labels = (user_ids.unsqueeze(0) == user_ids.unsqueeze(1)).float()
            mask = torch.eye(len(user_ids), device=user_ids.device)
            cl_loss = F.binary_cross_entropy_with_logits(sim[mask == 0], labels[mask == 0])
            result['cl_loss'] = cl_loss
        return result


# ====================== Dataset ======================

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


# ====================== 训练函数 ======================

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


def train_bt_contrastive(model, train_dl, device, cl_weight=0.1, temperature=0.1, epochs=1, lr=1e-3):
    """BT Contrastive: 同一 business_type 的样本拉近，不同 BT 的推开"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for ep in range(epochs):
        for bi, (features, labels, bt_ids, user_ids) in enumerate(train_dl):
            features = {k: v.to(device) for k, v in features.items()}
            bt = bt_ids.to(device)

            # forward
            reprs = model.get_repr(features)  # [B, 128]
            reprs = F.normalize(reprs, dim=-1)

            # BT 对比 loss
            bt_sim = reprs @ reprs.T  # [B, B]
            bt_sim = bt_sim / temperature
            bt_labels = (bt.unsqueeze(0) == bt.unsqueeze(1)).float()
            mask = torch.eye(len(bt), device=device)
            bt_cl = F.binary_cross_entropy_with_logits(bt_sim[mask == 0], bt_labels[mask == 0])

            # CTR loss
            logits = model.head(model.net[-1](reprs))
            pred = torch.sigmoid(logits).squeeze(-1)
            ctr_loss = F.binary_cross_entropy(pred, labels.to(device))

            loss = ctr_loss + cl_weight * bt_cl
            opt.zero_grad(); loss.backward(); opt.step()


def train_augment_contrastive(model, train_dl, device, cl_weight=0.1, temperature=0.1,
                               dropout_rate=0.3, epochs=1, lr=1e-3):
    """Augment Contrastive: 对每个样本 dropout 部分特征，两个 view 做对比"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    str_features = model.str_features
    n_str = len(str_features)

    for ep in range(epochs):
        for bi, (features, labels, bt_ids, user_ids) in enumerate(train_dl):
            features = {k: v.to(device) for k, v in features.items()}

            # View1: 原始特征
            repr1 = model.get_repr(features)
            repr1 = F.normalize(repr1, dim=-1)

            # View2: dropout 部分字符串特征
            feat_drop = np.random.rand(n_str) < dropout_rate
            aug_feat = dict(features)  # 先拷贝所有特征（包含数值特征）
            for i, f in enumerate(str_features):
                if feat_drop[i]:
                    aug_feat[f] = torch.zeros_like(features[f])  # mask 成 0
            repr2 = model.get_repr(aug_feat)
            repr2 = F.normalize(repr2, dim=-1)

            # 对比 loss: view1 vs view2
            sim = repr1 @ repr2.T
            sim = sim / temperature
            labels_cl = torch.arange(len(features[str_features[0]]), device=device)
            cl_loss = F.cross_entropy(sim, labels_cl) + F.cross_entropy(sim.T, labels_cl)

            # CTR loss
            logits = model.head(model.net[-1](repr1))
            pred = torch.sigmoid(logits).squeeze(-1)
            ctr_loss = F.binary_cross_entropy(pred, labels.to(device))

            loss = ctr_loss + cl_weight * cl_loss
            opt.zero_grad(); loss.backward(); opt.step()


def train_mlora(model, train_dl, device, cl_weight=0.0, temperature=0.1, epochs=1, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for ep in range(epochs):
        for bi, (features, labels, bt_ids, user_ids) in enumerate(train_dl):
            features = {k: v.to(device) for k, v in features.items()}
            bt_d = bt_ids.to(device)
            out = model(features, bt_d, user_ids.to(device), temperature=temperature)
            loss = F.binary_cross_entropy(out['pred'], labels.to(device))
            if cl_weight > 0:
                loss = loss + cl_weight * out.get('cl_loss', torch.tensor(0.0).to(device))
            opt.zero_grad(); loss.backward(); opt.step()


# ====================== 主流程 ======================

def main():
    print("="*60)
    print("data_raw 全实验（共10个配置）")
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

    # 10 个实验配置
    experiments = [
        # (name, model_fn, train_func, use_mlora, config_dict)
        ('baseline', lambda: BaselineCVR(str_features, num_features, str_vocab_sizes, 32, [512,256,128]),
         lambda m: train_baseline(m, train_dl, DEVICE, epochs=1), False, {}),

        ('bt_contrastive_cw0.1',
         lambda: BaselineCVR(str_features, num_features, str_vocab_sizes, 32, [512,256,128]),
         lambda m: train_bt_contrastive(m, train_dl, DEVICE, cl_weight=0.1, epochs=1), False,
         {'cl_type': 'bt_contrastive', 'cl_weight': 0.1}),

        ('bt_contrastive_cw0.05',
         lambda: BaselineCVR(str_features, num_features, str_vocab_sizes, 32, [512,256,128]),
         lambda m: train_bt_contrastive(m, train_dl, DEVICE, cl_weight=0.05, epochs=1), False,
         {'cl_type': 'bt_contrastive', 'cl_weight': 0.05}),

        ('augment_contrastive_cw0.1',
         lambda: BaselineCVR(str_features, num_features, str_vocab_sizes, 32, [512,256,128]),
         lambda m: train_augment_contrastive(m, train_dl, DEVICE, cl_weight=0.1, epochs=1), False,
         {'cl_type': 'augment_contrastive', 'cl_weight': 0.1}),

        ('mlora_rank4',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 4, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.0, epochs=1), True, {}),

        ('mlora_rank8',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 8, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.0, epochs=1), True, {}),

        ('mlora_rank12',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 12, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.0, epochs=1), True, {}),

        ('user_contrastive',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 4, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.1, epochs=1), True,
         {'cl_type': 'user_contrastive', 'cl_weight': 0.1}),

        ('mlora_rank4_user_cl0.1',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 4, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.1, epochs=1), True,
         {'cl_type': 'user_contrastive', 'cl_weight': 0.1}),

        ('mlora_rank8_user_cl0.1',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 8, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.1, epochs=1), True,
         {'cl_type': 'user_contrastive', 'cl_weight': 0.1}),

        ('mlora_rank12_user_cl0.1',
         lambda: MLoRACVR(str_features, num_features, str_vocab_sizes, 32, 128, n_domains, 12, [256,128]),
         lambda m: train_mlora(m, train_dl, DEVICE, cl_weight=0.1, epochs=1), True,
         {'cl_type': 'user_contrastive', 'cl_weight': 0.1}),
    ]

    all_results = {}
    for name, model_fn, train_fn, use_mlora, cfg in experiments:
        print(f"\n{'='*60}")
        print(f"训练: {name}")
        print(f"{'='*60}")
        model = model_fn().to(DEVICE)
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        train_fn(model)
        auc, preds, labels, bt_ids = evaluate(model, test_dl, DEVICE, use_mlora=use_mlora)
        bt_res = per_bt_eval(preds, labels, bt_ids, bt_encoder)
        print(f"Overall AUC: {auc:.4f}")
        all_results[name] = {'overall_auc': round(auc, 4), 'per_bt': bt_res, 'config': cfg}

    # 保存结果
    with open(f'{RESULTS_DIR}/data_raw_all_exp_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 生成 markdown 对比表格
    bt_order = list(bt_encoder.classes_)
    names = list(all_results.keys())
    header = "| BT | " + " | ".join(names) + " |"
    sep = "|------|" + "|------:" * len(names) + "|"
    rows = []
    for bt in bt_order:
        aucs = []
        for n in names:
            r = all_results[n]['per_bt'].get(bt, {})
            aucs.append(f"{r.get('auc', '-'):.4f}" if 'auc' in r else '-')
        rows.append(f"| {bt} | " + " | ".join(aucs) + " |")
    overall_row = "| **Overall** | " + " | ".join([f"**{all_results[n]['overall_auc']:.4f}**" for n in names]) + " |"

    md = f"""# data_raw 全实验结果 V2（10个配置）

## 配置
- embed_dim=32, hidden=[512,256,128], 1 epoch, batch_size={BS}
- 对比学习 temperature=0.1, augment dropout_rate=0.3

## per-BT AUC 对比

{header}
{sep}
""" + "\n".join(rows) + f"\n{overall_row}\n"
    with open(f'{RESULTS_DIR}/data_raw_all_exp_v2.md', 'w') as f:
        f.write(md)

    # 打印汇总
    print(f"\n{'='*60}")
    print("结果汇总")
    print(f"{'='*60}")
    print(header)
    print(sep)
    print(overall_row)

    print(f"\n结果已保存: {RESULTS_DIR}/data_raw_all_exp_v2.json")


if __name__ == '__main__':
    main()